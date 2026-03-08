import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from tqdm import tqdm

BASE_DATA_DIR = "/home/zayn/projects/SegVol_Project/data_preprocessed"

def generate_all_previews():
    # 抓取所有產出的 label 檔案
    label_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_label.nii.gz")))
    
    print(f"開始為 {len(label_files)} 個標籤產出預覽...")

    for lbl_path in tqdm(label_files, desc="Processing Previews"):
        fname = os.path.basename(lbl_path)
        # 預期格式: Segmentation_0324_1250_label.nii.gz
        parts = fname.split('_')
        date_str = parts[1]
        pid = parts[2]
        
        img_path = lbl_path.replace("_label.nii.gz", "_img.nii.gz")
        out_preview = f"preview_{date_str}_{pid}_check.png"

        if not os.path.exists(img_path): continue

        img_obj = nib.load(img_path)
        lbl_obj = nib.load(lbl_path)
        img_data = img_obj.get_fdata()
        lbl_data = lbl_obj.get_fdata()
        
        # 尋找有標籤的中心層
        z_indices = np.where(np.any(lbl_data > 0, axis=(0, 1)))[0]
        slice_idx = int(np.median(z_indices)) if len(z_indices) > 0 else img_data.shape[2] // 2

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        plt.title(f"CT: {date_str}_{pid}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        mask = lbl_data[:, :, slice_idx].T
        masked = np.ma.masked_where(mask == 0, mask)
        plt.imshow(masked, cmap='jet', alpha=0.5, origin='lower', interpolation='nearest')
        plt.title("Alignment Check")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(out_preview, dpi=120)
        plt.close()

if __name__ == "__main__":
    generate_all_previews()
    print("\n[Done] 所有預覽圖已產出！")