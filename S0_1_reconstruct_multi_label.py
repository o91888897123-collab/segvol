import os
import glob
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm
import cv2
import re

# --- 設定路徑 ---
DONE_ROOT = "/home/zayn/projects/mask/data/Done"
BASE_DATA_DIR = "/home/zayn/projects/SegVol_Project/data_preprocessed"
Z_OFFSET = 8 

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def fuse_all_batches():
    # 1. 找到 Done 底下所有符合 Segmentation_XXXX 的資料夾
    batch_folders = [d for d in os.listdir(DONE_ROOT) if d.startswith("Segmentation_")]
    
    ORGAN_MAP = {
        "mask_LV": 1, "mask_RV": 2, "mask_LA": 3, "mask_RA": 4, 
        "mask_AO": 5, "mask_PA": 6, "mask_PV": 7
    }

    print(f"找到 {len(batch_folders)} 個日期批次: {batch_folders}")

    for batch in batch_folders:
        batch_path = os.path.join(DONE_ROOT, batch)
        # 取得日期部分 (例如 0324)
        date_str = batch.split('_')[-1] 
        
        # 取得該批次下所有患者 ID
        patient_ids = sorted([d for d in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, d))])
        
        for pid in tqdm(patient_ids, desc=f"Batch {date_str}"):
            png_root = os.path.join(batch_path, pid)
            # 對齊檔名規則: Segmentation_{date}_{pid}_img.nii.gz
            ct_path = os.path.join(BASE_DATA_DIR, f"Segmentation_{date_str}_{pid}_img.nii.gz")
            out_path = os.path.join(BASE_DATA_DIR, f"Segmentation_{date_str}_{pid}_label.nii.gz")

            if not os.path.exists(ct_path):
                continue

            ref_img = nib.load(ct_path)
            nx, ny, nz = ref_img.shape
            final_mask_3d = np.zeros((nx, ny, nz), dtype=np.uint8)

            organ_folders = [d for d in os.listdir(png_root) if os.path.isdir(os.path.join(png_root, d))]
            for folder_name in organ_folders:
                organ_id = next((v for k, v in ORGAN_MAP.items() if k in folder_name), None)
                if organ_id is None: continue
                
                png_folder = os.path.join(png_root, folder_name)
                png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")), key=natural_sort_key)
                
                for z_idx, f in enumerate(png_files):
                    actual_z = z_idx + Z_OFFSET
                    if actual_z >= nz: break 
                    
                    # 使用益版確認過的最穩空間對齊邏輯
                    slice_data = np.flipud(np.array(Image.open(f).convert("L")))
                    if slice_data.shape[1] != nx or slice_data.shape[0] != ny:
                        slice_data = cv2.resize(slice_data, (nx, ny), interpolation=cv2.INTER_NEAREST)
                    
                    final_mask_3d[:, :, actual_z][slice_data.T > 0] = organ_id

            nib.save(nib.Nifti1Image(final_mask_3d, ref_img.affine, ref_img.header), out_path)

if __name__ == "__main__":
    fuse_all_batches()
    print("\n[Done] 所有批次重組完成！")