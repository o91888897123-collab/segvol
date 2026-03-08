# 因為 debug_plots 做出來會有品質參差不齊的問題
# 所以 recheck 品質過低的樣本
# 篩選出來後再研究是什麼問題導致的

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定路徑
EMBED_DIR = "./output/embeddings"
REPORT_DIR = "./output/quality_report"
os.makedirs(REPORT_DIR, exist_ok = True)

def check_quality():
    print(f"目前目錄:{os.getcwd()}")
    
    if not os.path.exists(EMBED_DIR):
        print(f"找不到資料夾 '{EMBED_DIR}'，請確認路徑是否正確。")
        return
    
    npy_files = [f for f in os.listdir(EMBED_DIR) if f.endswith("_regional_feat.npy")]
    
    # 確認有沒有找到檔案
    print(f" 在 '{EMBED_DIR}' 中找到了 {len(npy_files)} 個特徵檔案。")
    stats = []
    
    for f in npy_files:
        patient_id = f.replace("_regional_feat.py", "")
        data = np.load(os.path.join(EMBED_DIR, f))
        # 把二進位的檔案轉回 NumPy 矩陣
        # 形狀 (8, 768)
        
        #1 計算特徵總能量 (L2 Norm 的平均) 
        # (L2 Norm)^2 的定義與物理上能量的定義相同
        # 數值越低代表影像越模糊(特徵點越少)
        energy = np.linalg.norm(data, axis = 1).mean()    
        # np.linalg, Linear Algebra: 線性代數工具箱
        # Norm: 把每一個特徵直平分 相加 再開根號
        # axis = 1: 橫著算(每一行)  0->直  1->橫
        # 橫著算才可以從 8 個區域去對應各自的特徵強度值 
        
        #2 計算 8 個區域間的變異度
        spatial_varience = np.std(data, axis = 0).mean()
        # axis = 0: 從 8 個區域去看區域間的特徵強度差距
        # 如果單一區域的變異量趨近於 0 ，代表沒有層次 -> 無效影像
        
        stats.append({
            "patient_id": patient_id,
            "feature_energy": energy,
            "spatial_varience": spatial_varience
        })
        
    df = pd.DataFrame(stats)
        
    # 使用 IQR 判斷離群值
    q1 = df["feature_energy"].quantile(0.25)
    q3 = df["feature_energy"].quantile(0.75)
    iqr = q3 - q1
    # 做出壞樣本的區間: 低於(Q1 - 1.5 * IQR)
    energy_threshold = q1 - 1.5 * iqr
        
    # 標記潛在的壞數據
    df["bad"] = df["feature_energy"] < energy_threshold
    bad_list = df[df["bad"] == True]["patient_id"].tolist()
    # 把低於閾值的壞數據抓出來 並且把 ID 加入 list 中
        
    # 產出圖表
    plt.figure(figsize = (10, 6))
    plt.scatter(range(len(df)), df["feature_energy"], c = df["bad"].map({True:"red", False:"blue"}))
    plt.axhline(y = energy_threshold, color = "r", linestyle = "--", label = "Potential Bad Data Threshold")
    plt.title("Feature Energy Distribution")
    plt.xlabel("Patient Index")
    plt.ylabel("Mean Feature Energy")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "quality_check_plot.png"))
    plt.close()
        
    # 圖表存成 csv
    df.to_csv(os.path.join(REPORT_DIR, "quality_report.csv"), index = False)
    print(f"處理完成")
    print(f"總樣本數:{len(df)}")
    print(f"疑似異常樣本總數:{len(bad_list)}")
    if bad_list:
        print(f"檢查:{bad_list}")
            
if __name__ == "__main__":
    check_quality()
    
'''
造成異常的可能原因:
1. 掃描時心臟跳動造成重影 導致 Encoder 提取過多雜訊
2. preprocess 的資訊稀釋: 
    a. ROI 比例問題: 4 個人的心臟再切出的 128*128*128 方塊中佔比太小
    b. 計算區域平均特徵時 過多的 "0" 稀釋掉 心臟組織的 "1" 
3. 解剖構造的異質性
4. 數據在轉檔時 or 座標的偏移  
'''    