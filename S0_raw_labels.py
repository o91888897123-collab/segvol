from PIL import Image
import numpy as np

# 隨機選一張 PNG
path = "/home/zayn/projects/mask/data/Done/Segmentation_0324/1250/mask_LV/mask_0064.png"
img = Image.open(path)
data = np.array(img)
print(f"標籤數值包含: {np.unique(data)}") 
# 如果看到 [0 255] 純白mask