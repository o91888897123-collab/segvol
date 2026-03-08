import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from torch.nn import functional as F
from monai import transforms
from segvol_toolbox import get_segvol_model, calculate_soft_dice, get_official_transform, perform_ttest, get_automatic_prompt, get_text_embedding
from layer_module.heart_layers import HeartFC_v1
from tqdm import tqdm  # 進度條

#1 模型調用(from segvol_toolbox)
model, device = get_segvol_model()
fc_head = HeartFC_v1(input_dim=6144).to(device)
fc_head.eval()
# .eval():用來開啟 "考試" 功能

# 調用影像大小邏輯模組
official_transforms = get_official_transform()

#2 實驗數據載入
# 指向預處理後的資料夾
DATA_DIR = "./data_preprocessed"
OUTPUT_DIR = "./output"
df_quality = pd.read_csv("./output/quality_report/quality_report.csv")

HEART_MAP = {    
    "left ventricle": 1 ,  # 左心室
    "right ventricle": 2,  # 右心室
    "left atrium": 3,  # 左心房
    "right atrium": 4,  # 右心房
    "aorta": 5,  # 主動脈
    "pulmonary artery": 6,  # 肺動脈
    "pulmonary vein": 7,  # 肺靜脈
}
results = []
print(f"開始對{len(df_quality)}位病人進行效能評估")

pbar = tqdm(df_quality.iterrows(), total = len(df_quality), desc = "進度")

# 手動指定 Tensor 的格式
pixel_mean_t = torch.Tensor([123.675]).to(device).view(1, 1, 1, 1)
pixel_std_t = torch.Tensor([58.395]).to(device).view(1, 1, 1, 1)

for _, row in pbar:
    # _ : 放不需要資訊的垃圾桶
    # row: df_quality 中的內容
    pid = row["patient_id"]
    is_bad = row["bad"]
    
    # 去除附檔名 與 多餘後綴
    clean_id = pid.split("regional")[0].rstrip('_').strip() # .strip(): 用來去掉字串頭尾的空白字元
    
    # 對齊 S1 產出的檔案格式 (_img 與 _label)
    img_path = os.path.join(DATA_DIR, f"{clean_id}_img.nii.gz")
    gt_path = os.path.join(DATA_DIR, f"{clean_id}_label.nii.gz")
    
    # 如果 S1 沒產出該病人的檔案就跳過
    if not os.path.exists(img_path) or not os.path.exists(gt_path):
        continue

    try:
        # 使用 toolbox 定義的 get_official_transform
        # EnsureChannelFirstd 與 Resize(256, 256, 128)
        data_dict = official_transforms({"image":img_path, "label":gt_path})
        
        # 轉為 Tensor 推上 GPU 並加上 Batch 變成 5D [1, 1, 256, 256, 128]
        img_tensor = data_dict["image"].to(device)
        gt_tensor = data_dict["label"].to(device)
               
        # # 影像標準化 (對齊 S0_1)
        # img_tensor = (img_tensor - pixel_mean_t) / pixel_std_t
               
        img_tensor = torch.flip(img_tensor, dims=[2]) # 對 Height 維度進行翻轉
        
        # 因為 SegVol 內建 hard set RGB 三通道
        # 所以如果 img_tensor 為單通道則複製 
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1, 1)
                
        img_tensor = img_tensor.unsqueeze(0) # [1, 3, 256, 256, 256]
        gt_tensor = gt_tensor.unsqueeze(0)  # [1, 1, 256, 256, 256]
         # .unsqueeze(0) 兩次: 把形狀擴張成 5D 
        
        # 直接使用原廠模組
        with torch.no_grad():
            # 準備進入 Encoder 
            input_for_encoder = img_tensor[:, 0:1, ...]
            # 直接呼叫 Image Encoder 提取特徵 by ImageEncoderViT
            features = model.image_encoder(input_for_encoder)
        
            # 確保 features 不是 tuple
            if isinstance(features, (list, tuple)):
                features = features[0]
                
            # 跟 S2 一樣要調整維度位置成 [Batch, "Channel", Height, Width, Depth]
            # 因為這裡是直接套用 SegVol 原版模型
            
            # 維度變化:
            # Step1: 影像 input
            # [1, 1, 256, 256, 128] --> 5D  [Batch, Channel, H, W, D]
            
            # Step2: Patch Embedding
            # SegVol 的 Image Encoder 會以 16x16x16 一個 patch 讀取
            
            # Step3: Transformer 輸出的一條線 (把 5D 壓平成 3D) --> [1(Batch), 2048(Tokens), 768(Embedding)]
            # Step4: .view() 把 2048 塊積木塞回 16x16x8 的盒子裡
            # Step5: .permute() 把 channel 維度移到 shape[1]
            # Step6: Mask Decoder 會把 features 放大成[1, 1, 64, 64, 32]，再用三線插值放大回原始圖像尺寸
            if features.dim() == 3:  # if 3D
                #2048 tokens = 16 (H) * 16 (W) * 8 (D)
                b, n, c = features.shape
                # 確保 8 (深度 patch) 在最後面
                features = features.reshape(b, 16, 16, 8, c).permute(0, 4, 1, 2, 3).contiguous()
                # 此時 features 為 [B, 768, 16, 16, 8]
            
            # 建立合成 Mask & Dice 暫存處
            full_pred_mask = torch.zeros_like(gt_tensor) # [1, 1, 256, 256, 128]
            temp_dice_results = {}
            
            target_spatial_size = (tuple([int(s) for s in gt_tensor.shape[-3:]]))

            
            # 遍歷器官 提供文字提示
            for organ_name, organ_id in HEART_MAP.items():
                text_embed = get_text_embedding(model, [organ_name], device)
                
                # get_automatic_prompt 會執行mean(dim=(2, 3, 4))
                auto_sparse, auto_dense = get_automatic_prompt(model, features, text_embedding=text_embed)
            
                # 獲取位置編碼 (Positional Embedding)
                # 因為 PE 不在 model 裡 所以到 prompt_encoder 找
                if hasattr(model, "prompt_encoder"):
                    pos_embed = model.prompt_encoder.get_dense_pe()
                else:
                    pos_embed = model.get_dense_pe()

                # 如果 pos_embed 的[H, W, D] 跟 features 不一樣就強制縮放
                # [Batch, Channel, Height, Width, Depth] 從後面開始取三個
                if pos_embed.shape[-3: ] != features.shape[-3: ]:
                    # 把 PE 縮放到跟 Features 一樣的大小 
                    pos_embed = F.interpolate(
                    pos_embed,
                    size = features.shape[-3:],
                    mode = "trilinear",  # 用三線插值法
                    align_corners = False  # SegVol 原廠模型設定就是 False
                )
                                
                # 生成 Mask by MaskDecoder
                # MaskDecoder 通常回傳一個三元素的 tuple
                # (Mask, IoU Score, Low-res Masks)
                outputs = model.mask_decoder(
                    image_embeddings = features,
                    image_pe = pos_embed, 
                    # PE: Position Embeddings(位置編碼)
                    # model.get_dense_pe(): 在 segment_anything_volumetric/modeling 中的 sam.py
                    sparse_prompt_embeddings = auto_sparse, # sparse: 疏
                    dense_prompt_embeddings = auto_dense,   # dense: 密   #表示不管疏還是密 都不會給提示 
                    multimask_output = False,  
                    text_embedding = text_embed     
                    )
                low_res_masks = outputs[0]
                
                # 計算 Soft Dice 
                single_dice = calculate_soft_dice(low_res_masks, gt_tensor, {organ_name : organ_id})
                temp_dice_results.update(single_dice)
                
                # 放大回 256x256x128 進行合成
                organ_logits = F.interpolate(
                    low_res_masks, 
                    size =target_spatial_size,
                    mode = "trilinear",
                    align_corners = False
                )
                # 填入對應編號
                full_pred_mask[organ_logits > 0] = organ_id
                           
            # 區域特徵判斷 (FC layer)
            # 把 3D 特徵圖壓平成 6144為向量 (768 * 8 個區域)
            flatten_feat = features.reshape(features.shape[0], -1)
            # .view:用來變更包裝形狀 (總數不能改變)
            # -1: 表示前面的值給完後 自己算出後面應該要是多少            
            #　image_embeddings.size = (Batch, 8, 768) --> flatten_feat = (Batch_size, Total_features = 8x768)
            
            # 只取前 6144 個維度進行預測 ( : 6144)
            prediction = fc_head(flatten_feat[:, :6144])  
            # 第一個":" 代表 Batch_size 全部都要
            # 第二個":" 代表 Total_Features 只切 1~6144
        
        
        row_stat = {
            "patient_id": clean_id,
            "is_bad": is_bad,
            "fc_score": prediction.item(),
            "Mean_Dice":np.mean(list(temp_dice_results.values()))            
        }
        row_stat.update(temp_dice_results)  
        results.append(row_stat)
        
        # 1. 取得原始影像物件以獲得原始維度 (nx, ny, nz)
        orig_img_obj = nib.load(img_path)
        orig_nz_size = tuple([int(s) for s in orig_img_obj.shape])
        
        final_recorded_mask = full_pred_mask.permute(0, 1, 3, 4, 2)
        
        # 2. 放大回 Z 軸深度(256)
        final_save_mask = F.interpolate(
            final_recorded_mask.float(), # for 支援插值
            size = orig_nz_size,
            mode = "nearest",
        )
        pred_numpy = final_save_mask.detach().cpu().numpy()[0, 0, ...]
        
        # 3. 封裝成 NIfTI 並儲存 (使用與 GT 相同的空間資訊)
        pred_nii = nib.Nifti1Image(
            pred_numpy.astype(np.uint8), 
            orig_img_obj.affine, 
            orig_img_obj.header
        )
        
        nib.save(pred_nii, os.path.join(OUTPUT_DIR, f"{clean_id}_pred.nii.gz"))
        
        pbar.set_postfix({
            "PID":clean_id,
            "MeanDice": f"{row_stat['Mean_Dice']:.4f}"
        }) 
        torch.cuda.empty_cache()
        
    except Exception as e:
        tqdm.write(f"病人{clean_id} 錯誤: {e}") 
        
#3 數據整理 & 47 vs 51 對照實驗統計
if not results:
    print("\n 失敗：沒有病人成功跑完推論，請檢查預處理資料。")
else:
    final_df = pd.DataFrame(results)  # 全部資料都納入

    # 計算全體組別
    all_mean_dice = final_df['Mean_Dice'].mean()
    all_std_dice = final_df['Mean_Dice'].std()

    # 計算優質組
    clean_group = final_df[final_df['is_bad'] == 0]
    clean_mean_dice = clean_group['Mean_Dice'].mean()
    clean_std_dice = clean_group['Mean_Dice'].std()

    # 計算效能提升的比率
    improvement = ((clean_mean_dice - all_mean_dice) / all_mean_dice) * 100

    #4 P-value 檢定
    # 兩組的 Dice 比較
    t_stat, p_val, sig_result = perform_ttest(clean_group['Mean_Dice'], final_df['Mean_Dice'])

    #5 terminal report
    print("\n" + "="*40)
    print(f"分割實驗report(成功樣本 N={len(final_df)}):")
    print(f"全體樣本 Dice:{all_mean_dice:.4f}±{all_std_dice:.4f}")
    print(f"優質樣本 Dice:{clean_mean_dice:.4f}±{clean_std_dice:.4f}")
    print(f"排除異常樣本後，效能提升:{improvement:.3f}%")
    print(f"統計檢定 p-value:{p_val:.6f}")
    print(f"差異顯著性:{sig_result}")

    #5 outcome storage
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    # report(1): 數據清單("dice", "FC score", "quality_label" of each patient)
    detailed_path = os.path.join(OUTPUT_DIR, "detailed_performance_results.csv")
    final_df.to_csv(detailed_path, index = False)

    # report(2): 實驗結論摘要
    summary_data = {
        "指標":[
            f"全體樣本(N = {len(final_df)})", 
            f"優質樣本(N = {len(clean_group)})", 
            "效能提升(%)", 
            "統計顯著性(p-value)"
        ],
        "平均 Dice 分數":[
            f"{all_mean_dice:.4f}",
            f"{clean_mean_dice:.4f}",
            f"{improvement:.3f}%", 
            f"{p_val:.6f}"
        ],
        "標準差(STD)":[
            f"{all_std_dice:.4f}",
            f"{clean_std_dice:.4f}",
            "-",
            "-"
        ],
        "備註":[
            "包含所有成功處理樣本",
            "僅含 bad=0 樣本",
            f"提升幅度 {improvement:.2f}%",
            sig_result
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary_report.csv")
    summary_df.to_csv(summary_path, index = False, encoding = "utf-8-sig")

    print(f"\n 詳細結果已存入:{detailed_path}")
    print(f"結論摘要已存入:{summary_path}")
    
# 找出失蹤的 2 個人
expected_ids = set(df_quality["patient_id"].apply(lambda x: x.split("regional")[0].rstrip('_').strip()))
success_ids = set(final_df["patient_id"].apply(lambda x: x.split("regional")[0].rstrip('_').strip()))
missing_ids = expected_ids - success_ids

print(f"\n 數據搜尋結果：")
print(f"失蹤的病人 ID: {missing_ids}")