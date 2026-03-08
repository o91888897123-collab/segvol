import os
import nrrd
import nibabel as nib
import numpy as np
import glob
from tqdm import tqdm # 進度條
from skimage.transform import resize

# path setting
# 原始資料引入 (指向你發現原始檔所在的 projects/mask)
ROOT_DIR = "/home/zayn/projects/mask/data/Done"
# 預處理後的資料輸出到目標資料夾中 (SegVol 專案下)
OUTPUT_DIR = "/home/zayn/projects/SegVol_Project/data_preprocessed"

# 如果沒有資料夾就自動建立並存入
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 設定目標尺寸為(256, 256, 256) 避免 Z 軸壓縮
TARGET_SHAPE = (256, 256, 256)

def process_patient(patient_path):
    # 從路徑中提取病人ID
    patient_id = os.path.basename(patient_path)
    batch_name = os.path.basename(os.path.dirname(patient_path))
    
    # 搜尋標註檔
    # 第一個*:開頭可以是任何數只要檔名有包含Segmentation即可
    # 第二個*:在Segmentation之後、副檔名之前可以是任何字
    # .nrrd:副檔名必須是 .nrrd
    label_files = glob.glob(os.path.join(patient_path, "*Segmentation*.nrrd"))
    
    # 定義影像檔路徑 (例如: 1250.nrrd)
    img_file_path = os.path.join(patient_path, f"{patient_id}.nrrd")
    
    # 確保影像檔 (字串) 與 標註檔 (清單) 都存在才執行
    if not os.path.exists(img_file_path) or not label_files:
        return f"{patient_id} 檔案不齊全"

    try:
        #1 讀取 data 和 header (同時讀取影像與標註)
        data_label, header = nrrd.read(label_files[0])
        img_data, _ = nrrd.read(img_file_path)
        
        #2 Windowing (窗寬窗位處理)
        # 把位置限制在 -200 ~ 500 才能讓心臟位置凸顯
        img_data = np.clip(img_data, -200, 500)
        # 官方模型預期區間為 0 ~ 1
        img_data = (img_data - (-200)) / (500 - (-200))
        
        #3 找出有mask的區域(Bounding Box)
        coords = np.array(np.where(data_label > 0)) 
        # np.where(data > 0) 會找出所有標註像素(點)的 [x, y, z] 座標  
        
        # 如果 mask 檔全是 0 就跳過   
        if coords.size == 0: return f"{patient_id} 為空值" 
        
        #4 設定邊距和裁切的範圍
        # Padding: 在心臟 mask 的邊界外多留 10 像素 避免邊緣被切得太死
        p = 10
        
        # 計算三個軸的 min 和 max 並且加上邊距
        x_m, x_M = max(0, coords[0].min()-p), min(data_label.shape[0], coords[0].max()+p)
        y_m, y_M = max(0, coords[1].min()-p), min(data_label.shape[1], coords[1].max()+p)
        z_m, z_M = max(0, coords[2].min()-p), min(data_label.shape[2], coords[2].max()+p)
        # 概念是 各軸的最小值範圍 assign 為 0 ~ mask的最小邊界再加邊距 p ，最大值範圍 assign 為 形狀邊界 ~ mask的最大邊界再加邊距 p
        
        #5 執行裁切 (ROI Extraction)
        # 為了運算速度把影像體積縮減 🌟 同時裁切影像與標註
        cropped_img = img_data[x_m : x_M, y_m : y_M, z_m : z_M]
        cropped_label = data_label[x_m : x_M, y_m : y_M, z_m : z_M]
        
        #6 Bicubic interpolate 和對齊
        # 文獻中影像用 order = 3 (Bicubic), 標籤用 order = 0 (Nearest)
        #並且統一為 256x256x256 讓模型在 Z 軸更準確
        final_img = resize(cropped_img, TARGET_SHAPE, order = 3, mode = "edge", anti_aliasing=True)
        final_label = resize(cropped_img, TARGET_SHAPE, order = 0, mode = "edge", anti_aliasing=False)
        """ 
        order = 0: 選擇 "最近的鄰近值" 填補多出來的空間 (因為標籤的值必須是整數 確保不會讓標籤變質)
        order = 3: 提供 Image 更多元的參考 而不是只看相鄰的像素
        mode = "edge": 當我們進行縮放或旋轉時 視窗會多出部分空白的邊界 edge 會直接拿最邊緣的像素質往外拉填補空白
        anti_aliasing: 將大影像縮小時邊緣會產生鋸齒狀
        -Image True: 會進行一次高斯模糊 標籤使用後會導致器官間的交界產生非整數的值 導致標籤變質
        """
        
        
        #7 Affine matrix 重新計算
        # 因為解析度變了 所以每個像素代表的實體間距 (Spacing) 也變了
        spacing_dirs = header.get("space direction", np.eye(3))
        orig_spacing = np.linalg.norm(spacing_dirs, axis = 1)
        # 計算縮放後矩陣的新間距: (原始裁切尺寸*原始間距) / 256
        new_spacing = (np.array(cropped_img.shape) * orig_spacing) / np.array(TARGET_SHAPE)
        
        origin = header.get("space origin", np.zeros(3)) # [0, 0, 0]
        affine = np.eye(4) 
        affine[:3, :3] = np.diag(new_spacing)  # 更新縮放比例
        affine[:3, 3] = origin
        """
                          origin
                            |
        np.eye(4)=[1, 0, 0, 0
                   0, 1, 0, 0
                   0, 0, 1, 0
                   0, 0, 0, 1] -前 3 個 0 放 origin 的 X Y Z 座標
        """
        
        #8 封裝並儲存為 NIfTI
        # 將裁切後的矩陣和訪設矩陣結合成一個 NIfTI 物件
        new_img_obj = nib.Nifti1Image(final_img.astype(np.float32), affine)
        new_label_obj = nib.Nifti1Image(final_label.astype(np.float32), affine)
        
        # 儲存影像檔：存檔名的格式指定為 "批次_病人ID_img.nii.gz"
        img_output = f"{batch_name}_{patient_id}_img.nii.gz"
        nib.save(new_img_obj, os.path.join(OUTPUT_DIR, img_output))
        
        # 儲存標註檔：存檔名的格式指定為 "批次_病人ID_label.nii.gz"
        label_output = f"{batch_name}_{patient_id}_label.nii.gz"
        nib.save(new_label_obj, os.path.join(OUTPUT_DIR, label_output))

        return f"{patient_id} 成功" #成功後回傳
    
    except Exception as e:
        return f"{patient_id} 錯誤: {e}"
    
# 先掃描 SegmentationXXXX (第一層) 再找裡面的病人資料夾(第二層)
all_folders = []

#使用 glob 尋找 Done/*/*
for path in glob.glob(os.path.join(ROOT_DIR, "*", "*")):
    if os.path.isdir(path):
        all_folders.append(path)

# any(c.isdigit() for c in f.name) 是為了確保資料夾名字有數字
folders = [f for f in all_folders 
            if any(c.isdigit() for c in os.path.basename(f))]

print(f"偵測到路徑: {ROOT_DIR}")
print(f"準備預處理 {len(folders)} 位病人")

# 啟動進度條功能
with tqdm(total=len(folders), desc="Preprocessing") as pbar:
    for folder in folders:
        # 執行單一病人的處理
        res = process_patient(folder)
        
        # 如果有回傳結果就顯示在進度條右邊
        if res: pbar.set_postfix_str(res)
        pbar.update(1)

print(f"\n 預處理完成 產出檔案已存至: {OUTPUT_DIR}")