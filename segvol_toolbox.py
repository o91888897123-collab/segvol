import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from monai import transforms
from scipy import stats
from transformers import CLIPTokenizer, CLIPTextModel
import clip

# 並異類別用來存放模型需要的參數
class SegVolArgs:
    def __init__(self):
        self.image_size = 256   # SegVol 預設的 3D 尺寸
        self.out_size = 256     # 輸出尺寸
        self.patch_size = 16    # Transformer 的 Patch 大小
        self.spatial_size = (128, 256, 256)
        
        self.encoder_embed_dim = 768 # feature dim
        self.encoder_depth = 12      # 層數
        self.encoder_num_heads = 12  # 多頭注意力機制數量
        self.encoder_global_attn_indexes = [2, 5, 8, 11]   # 全域注意力層索引
        
        self.prompt_embed_dim = 256
        self.checkpoint = None       # 預設為空值 由 build function 傳入

# 路徑與環境處理
def init_segvol_env():
    """自動搞定所有路徑與環境"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    segvol_package_root = os.path.join(project_root, 'SegVol')
    
    # 注入路徑
    if segvol_package_root not in sys.path:
        sys.path.insert(0, segvol_package_root)
    
    # 自動補辦身分證 (__init__.py)
    # (此處可加入我們之前的自動 touch 邏輯)
    return project_root, segvol_package_root

# 定義文字編碼器骨架，對齊權重掃描中的 model.text_encoder 結構
class SegVolTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用與 SegVol 相同的 CLIP 基礎模型
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        # 對齊權重表中的 [768, 512] 形狀
        self.dim_align = nn.Linear(512, 768)
        
    def forward(self, input_ids):
        outputs = self.clip_text_model(input_ids=input_ids)
        # 提取 [CLS] token 的特徵 (pooler_output)
        return self.dim_align(outputs.pooler_output)

# 模型調用
def get_segvol_model(ckpt_name="segvol_trained.pth"):
    """一鍵喚醒模型"""
    _, pkg_root = init_segvol_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 準備模型設定(傳給原廠 build 函式)
    model_cfg = SegVolArgs()
    
    # 定義權重路徑
    ckpt_path = os.path.join(pkg_root, "checkpoints", ckpt_name)
    
    # 從我們打通的路徑引用
    from segment_anything_volumetric.build_sam import build_sam_vit_3d
    
    # 建立空殼模型
    model = build_sam_vit_3d(checkpoint = None, args = model_cfg)
    
    model.text_encoder = SegVolTextEncoder()
    
    # 讀取權重檔 (確認裡面到底幾個)
    state_dict = torch.load(ckpt_path, map_location = "cpu")
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
        
        # 如果維度對不上，直接不載入該層，使用初始化權重
        if new_key in model.state_dict():
            if v.shape != model.state_dict()[new_key].shape:
                print(f"跳過維度不匹配層: {new_key} (Checkpoint: {v.shape}, Model: {model.state_dict()[new_key].shape})")
                continue 
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
        
    # 自動偵測全種檔的 " PE 值" 和 "長度"
    pe_key = "image_encoder.patch_embed.position_embeddings"
    if pe_key in new_state_dict:
        ckpt_pe = new_state_dict[pe_key] # [1, 2048, 768]
        if model.image_encoder.patch_embedding.position_embeddings.shape != ckpt_pe.shape:
            print(f"修正 PE 維度: {model.image_encoder.patch_embedding.position_embeddings.shape} -> {ckpt_pe.shape}")
            model.image_encoder.patch_embedding.position_embeddings = nn.Parameter(
                torch.zeros(ckpt_pe.shape)
            )
            model.image_encoder.patch_embedding.num_patches = ckpt_pe.shape[1]
    
    model.load_state_dict(new_state_dict, strict = False)
    print(f"正在載入模型至 {device}...")
    model.to(device)
    model.eval()
    return model, device

def get_automatic_prompt(model, image_features, text_embedding = None):
    """
    把 features 轉為解碼器的訊號
    """
    device = image_features.device
    # 把 [B, 768, 16, 16, 16] 的 image_features 壓平成全域向量 [1, 768]
    # 代表 "模型看到的整張心臟影像重點"
    global_feat = torch.mean(image_features, dim = (2, 3, 4))  # [B, 768]
    
    # 將 text_embedding 設為 None 傳入以符合函數定義
    # 解決 PromptEncoder.forward() missing 1 required positional argument 錯誤
    # 因為沒有 prompt 所以餵入 dummy 訊號啟動解碼器
    # 取得基礎提示 (256維)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points = None,
        boxes = None, 
        masks = None,
        text_embedding = text_embedding
    )
    
    # 動態對齊 不直接寫256
    target_dim = sparse_embeddings.shape[-1]
    
    # 把特徵截斷到模型預期得到的長度
    feat_to_add = global_feat.unsqueeze(1)[:, :, :target_dim]
    
    # 把 image_features 的資料融入 sparse_embeddings
    # Decoder 就會針對圖的特徵進行辨識
    # 做加法 [1, N, target_dim] + [1, 1, target_dim]
    auto_sparse = sparse_embeddings + feat_to_add
    # global_feat[:, :, 256] 會把 768 切掉只剩 256
    
    return auto_sparse, dense_embeddings
    
# Dice Score evaluation
def calculate_soft_dice(low_res_logits, gt_tensor, class_map):
    """
    parameter:
    -low_res_logits: 模型輸出的原始數值[1, C, 64, 64, 32] 
    -gt_tensor: 標準答案[1, 1, 256, 256, 128] 
                       [Batch, Channel, Depth, Height, Width] 數值為 0, 1, 2, 3...
    -class_map: 定義類別和標籤值{"LA":1, ....}
    
    return:
    -一個字典包含各類別的Soft Dice 分數
    """
    
    # hasattr(object, name-屬性): 如果 object 有該屬性就回傳 True
    if hasattr(low_res_logits, "as_tensor"):
        low_res_logits = low_res_logits.as_tensor()
    if hasattr(gt_tensor, "as_tensor"):
        gt_tensor = gt_tensor.as_tensor()
        
    dice_results = {}
    
    if low_res_logits.dim() == 4:
        low_res_logits = low_res_logits.unsqueeze(1)
    # 把解析度放大回原始尺寸 (避免小尺寸的微小誤差 回到原始尺寸後會被放大)
    # 使用 Trillinear 插值
    unsampled_logits = F.interpolate(
        low_res_logits, 
        size = gt_tensor.shape[-3:],  # 自動抓 (256, 256, 128)
        mode='trilinear', 
        align_corners=False
    )
    # 轉為機率 (Soft Dice) 
    # SegVol 用 Sigmoid 處理各通道
    probs = torch.sigmoid(unsampled_logits)
    
    # 如果模型輸出 3 個就取第一個通道
    # 要更精確就在 S4 選完 best_idx 再傳進來
    if probs.shape[1] > 1:
        probs = probs[:, 0:1, ...]
    
    dice_results = {}
    for class_name, label_idx in class_map.items():
        # 提取該器官的值(GT == 1, 2, 3...)
        target_gt = (gt_tensor == label_idx).float()
        
        # Soft Dice function: 2 * ( P * G ) / ( P + G )
        intersection = (probs * target_gt).sum()
        union = probs.sum() +target_gt.sum()
        
        dice = (2. * intersection + 1e-6) / ( union +1e-6 )
        dice_results[class_name] = dice.item()
        
    return dice_results

# 影像大小標準化
def get_official_transform():
    """
    醫學影像標準化流程：
    1. 確保有 Channel 維度
    2. 縮放至原廠指定的 256x256x128
    """
    return transforms.Compose([
        # 讀取影像路徑
        transforms.LoadImaged(keys=["image", "label"]),
        # 把 [D, H, W] 變成 [1, D, H, W]
        transforms.Orientationd(keys=["image", "label"], axcodes = "RAS"),
        # 調整尺寸，影像用三線性插值，標註用最近鄰插值（避免標註出現小數點）
        transforms.Resized(
            keys=["image", "label"], 
            spatial_size=(128, 256, 256),
            mode=("trilinear", "nearest")
        ),
        # 額外加上述值標準化 讓 SegVol 的特徵提取更穩定
        transforms.ScaleIntensityd(keys=["image"]),
    ])
    
# 獨立樣本 T 檢定
def perform_ttest(group_good, group_all):
    t_stat, p_value = stats.ttest_ind(group_good, group_all, equal_var = False)
    
    # 判斷是否顯著
    is_significant = "是(有顯著差異)" if p_value < 0.05 else "否(不具顯著差異)"
    
    return t_stat, p_value, is_significant

def load_clip_once(device):
    model, _ = clip.load("ViT-B/16", device = device)
    return model


# 在全域初始化一個 Tokenizer (SegVol 訓練時標準規格)
# 如果 Server 沒網路，可以手動下載 clip-vit-base-patch16 的 vocab.json
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

def get_text_embedding(model, text_list, device):
    """
    直接調用模型內部的 text_encoder 與 dim_align
    """
    if isinstance(text_list, str):
        text_list = [text_list]
        
    # 將文字轉換為模型看得懂的 ID
    inputs = tokenizer(text_list, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 直接呼叫模型內部的文字編碼模組
        return model.text_encoder(inputs.input_ids).float()