import os
import glob

# 1. 定位路徑
RAW_DATA_PATH = "/home/zayn/projects/mask/data/Done"
S1_OUTPUT_DIR = "./data_preprocessed"
S2_OUTPUT_DIR = "./output/embeddings"

def extract_id_from_filename(filename):
    # 🌟 強化版提取：假設檔名是 BatchName_ID_img.nii.gz
    # 我們取含有 'img' 或 'feat' 之前的那個部分
    parts = filename.split('_')
    if 'img.nii.gz' in filename:
        return parts[-2] # 倒數第二個通常是 ID
    if 'regional_feat.npy' in filename:
        return parts[-3] # 倒數第三個通常是 ID
    return None

# 2. 取得三方名單
raw_ids = []
for path in glob.glob(os.path.join(RAW_DATA_PATH, "*", "*")):
    if os.path.isdir(path) and any(c.isdigit() for c in os.path.basename(path)):
        raw_ids.append(os.path.basename(path))

s1_files = [os.path.basename(f) for f in glob.glob(os.path.join(S1_OUTPUT_DIR, "*_img.nii.gz"))]
s1_ids = [extract_id_from_filename(f) for f in s1_files]

s2_files = [os.path.basename(f) for f in glob.glob(os.path.join(S2_OUTPUT_DIR, "*_regional_feat.npy"))]
s2_ids = [extract_id_from_filename(f) for f in s2_files]

# --- 診斷輸出 ---
print(f"--- 🕵️ 數據偵探診斷中 ---")
print(f"📍 原始數據 ID 範例: {raw_ids[:2]}")
print(f"📍 S1 提取 ID 範例: {s1_ids[:2]}")
print(f"📍 S2 提取 ID 範例: {s2_ids[:2]}")
print("-" * 20)

# 轉成 set 進行比對
raw_set = set(raw_ids)
s1_set = set(s1_ids)
s2_set = set(s2_ids)

missing_in_s1 = raw_set - s1_set
ghost_in_s2 = s2_set - s1_set

print(f"✅ 原始總數: {len(raw_set)}")
print(f"✅ S1 成功產出: {len(s1_set)}")
print(f"✅ S2 成功產出: {len(s2_set)}")

if missing_in_s1:
    print(f"❌ 真正失蹤在 S1 的 ID ({len(missing_in_s1)}人): {missing_in_s1}")
else:
    print("🎉 S1 預處理全部過關！")

if ghost_in_s2:
    print(f"👻 S2 中的幽靈檔案 (存在 S2 但不存在 S1) ({len(ghost_in_s2)}人): {ghost_in_s2}")
    print("   👉 建議：請刪除 output/embeddings 資料夾後，重新執行 S2。")