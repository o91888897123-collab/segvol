import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import glob
from tqdm import tqdm
import nibabel as nib 
import matplotlib.pyplot as plt 

# 先確立當前路徑
current_dir = os.path.abspath(__file__) 
# cwd,current working directory
# abspath, absolute path
project_root = os.path.dirname(current_dir)

#1 將 SegVol 加入路徑確保能 import 子資料夾內的modeling
# SegVol是下載下來的資料夾 所以手動加入路徑
segvol_repo_path = os.path.join(project_root, "SegVol")
sys.path.append(segvol_repo_path)

if segvol_repo_path not in sys.path:
    sys.path.insert(0, segvol_repo_path)
# SegVol_Project/SegVol/segment_anything_volumetric/modeling

# 從 SegVol 原始碼中匯入模型類別
from segment_anything_volumetric.modeling.sam import Sam
from segment_anything_volumetric.modeling.image_encoder import ImageEncoderViT

# 設定參數
INPUT_DIR = "./data_preprocessed"
CKPT_PATH = "./SegVol/checkpoints/segvol_vit_b_batch8.pth"
OUTPUT_DIR = "./output/embeddings"
# DEBUG_DIR = "./output/debug_plots"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok = True)
# os.makedirs(DEBUG_DIR, exist_ok=True) # 建立驗證圖資料夾

# # 驗證圖
# def save_debug_plot(tensor, patient_id):
#     """
#     將處理後的 128x128x128 影像切片存出，確認 Orientation 是否正確轉為 RAS。
#     """
#     # 拿掉 Batch 和 Channel 維度，轉回 CPU numpy
#     img_data = tensor.squeeze().cpu().numpy()
#     mid = 128 // 2 # 取得中心切片
    
#     plt.figure(figsize=(15, 5))
#     # Axial 視角 (D 軸)
#     plt.subplot(1, 3, 1); plt.imshow(img_data[mid, :, :], cmap='gray'); plt.title('Axial (R-L)')
#     # Coronal 視角 (H 軸)
#     plt.subplot(1, 3, 2); plt.imshow(img_data[:, mid, :], cmap='gray'); plt.title('Coronal (A-P)')
#     # Sagittal 視角 (W 軸)
#     plt.subplot(1, 3, 3); plt.imshow(img_data[:, :, mid], cmap='gray'); plt.title('Sagittal (S-I)')
    
#     plt.suptitle(f"Verification - Patient: {patient_id} (Orientation: RAS)")
#     plt.savefig(os.path.join(DEBUG_DIR, f"{patient_id}_check.png"))
#     plt.close()
    
def load_model():
    # 手動建立 ViT 
    # 參數與SegVol 的 vit-b 一致
    image_encoder = ImageEncoderViT(
        img_size = 128,
        patch_size = 16,
        in_chans = 1,  # input channels = 1 (灰階)
        embed_dim = 768,
        depth = 12,  # 通過 12 層神經網路
        num_heads = 12,  # 多頭注意機制: 代表模型從 12 個不同的注意力角度去看心臟( 768/12 = 64, 每個頭負責關注64維的特徵)
        mlp_ratio = 4,   # 思考時模型會把 768 維放大四倍進行複雜計算 最後再縮回 768
        out_chans = 768
    )
    #初始化 Sam 類別
    model = Sam(
        # Vision Transformer, ViT: 負責看懂影像
        image_encoder = image_encoder,  
        
        # 不需要使用 prompt 的接受器 讓他自動掃描即可 
        prompt_encoder = None,
        
        # 不需要產出圖像 只要他的數值即可  # 可以嘗試可不可以 work(在prompt None情況下)
        mask_decoder = None,   
             
        # 像素平均值: 影像進入模型前會先執行 x_new = x_old - mean
        # 雖然影像是灰階 但是 Sam 的架構仍然是 R G B 三通道
        pixel_mean = [123.675, 116.28, 103.53],  
              
        # 數值波動的幅度: 執行完減法後會再執行 x_final = x_new/std
        # 用來統一對比度
        pixel_std = [58.395, 57.12, 57.375]
    )   
        # mean & std 數值由 "grep -r "pixel_mean" SegVol/segment_anything_volumetric/"抓取( mean 一樣) 
        
    # 載入權重
    # state_dict: 一個大型的清單裡面裝滿權重的參數數值
    # 藉由 CKPT_PATH 導入到 DEVICE 中
    state_dict = torch.load(CKPT_PATH, map_location = DEVICE)
        
    # 移除 "model."前綴 以匹配類別的變數名稱
    # 把權重字典裡面的 Key 名字開頭有 'model.' 的都去掉然後存入新字典中(for 標籤對齊)
    new_state_dict = {k.replace("model.", ""): v 
                      for k, v in state_dict.items()}
    
    # strict = False: 代表不用嚴格地把所有小零件都對上
    model.load_state_dict(new_state_dict, strict = False)
    
    # .to(DEVICE): 定位就緒 把 機器/模型 送進裝置中 (CPU/GPU)
    # .eval(): 評估功能 進入工作 / 考試狀態(關閉 training 模式)
    # Dropout 功能關閉 # 停止計算新的平均值 改用原始數據( ImageNet )提供的平均值
    model.to(DEVICE).eval()
    return model    

def main():
    model = load_model()
    # 搜尋 51 個 ROI 檔案 (包含.nii.gz的檔案)
    nii_files = glob.glob(os.path.join(INPUT_DIR, "*_img.nii.gz*"))
    
    print(f"開始提取{len(nii_files)}位病人心臟特徵")
    
    for f_path in tqdm(nii_files, desc = "Extracting"):
        full_id  = os.path.basename(f_path).replace("_img.nii.gz", "")
        
        try:
            #1 讀取 & 資料型態轉換
            img = nib.load(f_path)
            data = img.get_fdata()  # f: floating point # 把圖形中的浮點資料拿出來
            
            # S1 已經做過 Windowing 和 0~1 縮放 所以這裡轉成 Tensor
            input_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(DEVICE)  # 剛剛把模型傳進去了 現在換資料
            
            #2 把 256x256x256 原始尺寸 縮小到模型預期的 128x128x128
            # 使用 area 平均後提取縮小後的影像特徵
            input_tensor = F.interpolate(input_tensor, size = (128, 128, 128), mode = "area")
            # align_corners: 使用像素中心對齊 會避免整體影像被拉伸  # area 內建避免鋸齒形狀
            # F.interpolate: 重採樣工具(縮放)
            
            #ˇ 進入 ViT encoder 進行推論(不計算梯度)
            with torch.no_grad():
                # 直接呼叫 patch_embed 自動完成 PatchEmbed + PosEmbed
                # 將影像初步切塊並向量化 [1, 768, 8, 8, 8]
                x = model.image_encoder.patch_embed(input_tensor)
                if isinstance(x, list):
                    x= x[-1]
                
                # 不必再做維度的移轉 因為 model.image_encoder 內部會自主移動並在出 Encoder 前還原
                    
                # 區域提取: 立體八分切分 (全域平均池化可能會導致失去病灶辨認的資訊)
                regional_list = []
                # 將 8x8x8 的空間特徵圖切成 8 個 4x4x4 的區域
                for d in [0, 4]:
                    for h in [0, 4]:
                        for w in [0, 4]:
                            # 提取局部立方體: [1, 768, 4, 4, 4]
                            cube = x[:, :, d:d+4, h:h+4, w:w+4]
                            
                            # 對局部區域取平均  ( dim(2, 3, 4) )  的 [D', H', W']取平均值
                            # .cpu(): 把資料從 GPU 搬回 CPU 
                            # 最後轉成 numpy 以便後續統計
                            region_mean = torch.mean(cube, dim=(2, 3, 4))
                            regional_list.append(region_mean.squeeze().cpu().numpy())
                
                # 8個區域的特徵堆疊 最終形狀為 (8, 768)
                embedding_final = np.stack(regional_list) # (8, 768)
                
            #6 存成 .npy (feat for features)
            save_path = os.path.join(OUTPUT_DIR, f"{full_id}_regional_feat.npy")
            np.save(save_path, embedding_final)
            
        except Exception as e:
            import traceback
            print(f"病人{full_id}提取失敗: {e}")
            traceback.print_exc()  # 用於查看錯誤原因
            
if __name__ == "__main__":
    main()