import os
import glob
import nibabel as nib
import numpy as np

# --- 1. 自動偵測路徑 ---
gt_dir = "./data_preprocessed"
pred_dir = "./output"

print(f"=== 資料夾檢查 ===")
print(f"GT 資料夾 ({gt_dir}) 是否存在: {os.path.exists(gt_dir)}")
print(f"Pred 資料夾 ({pred_dir}) 是否存在: {os.path.exists(pred_dir)}")

# 列出前 3 個檔案當範例
if os.path.exists(gt_dir):
    print(f"GT 範例檔案: {os.listdir(gt_dir)[:3]}")
if os.path.exists(pred_dir):
    print(f"Pred 範例檔案: {os.listdir(pred_dir)[:3]}")

# --- 2. 嘗試自動配對患者 1250 ---
# 這裡我們用模糊搜尋，避免檔名差一個字就找不到
gt_search = glob.glob(os.path.join(gt_dir, "*1250*label.nii.gz"))
pred_search = glob.glob(os.path.join(pred_dir, "*1250*"))

if not gt_search or not pred_search:
    print("\n[錯誤] 找不到 1250 號病人的對應檔案。")
    print(f"搜尋到的 GT: {gt_search}")
    print(f"搜尋到的 Pred: {pred_search}")
else:
    gt_path = gt_search[0]
    pred_path = pred_search[0]
    print(f"\n成功配對！\nGT: {gt_path}\nPred: {pred_path}")

    # --- 3. 執行重疊度檢查 ---
    gt_obj = nib.load(gt_path)
    pred_obj = nib.load(pred_path)
    gt_data = gt_obj.get_fdata()
    pred_data = pred_obj.get_fdata()

    # 只要 > 0 就視為標籤
    gt_mask = gt_data > 0
    pred_mask = pred_data > 0
    
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    dice = (2.0 * intersection) / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) > 0 else 0

    print(f"\n=== 重疊度診斷 ===")
    print(f"GT 標籤數值: {np.unique(gt_data)}")
    print(f"Pred 標籤數值: {np.unique(pred_data)}")
    print(f"交集像素 (Intersection): {intersection}")
    print(f"3D Binary Dice: {dice:.6f}")

    if intersection == 0:
        print("\n[警告] 完全沒有交集！請檢查預覽圖，標籤可能上下顛倒了。")