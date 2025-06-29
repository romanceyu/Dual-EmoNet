import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from adjustText import adjust_text

# 1. 准备数据
models = [
    "VGG16", "VGG19", "ResNet18", "ResNet34", "ResNet50",
    "MobileNetV2", "DenseNet", "EfficientNet", "Dual_EmoNet"
]
params_list = np.array([61.26, 72.46, 11.18, 21.29, 25.95, 3.40, 7.25, 5.30, 42.49])
flops_list  = np.array([0.55,  0.69,  0.57,  1.17,  1.30, 0.32, 0.55, 0.39, 0.46])
accuracy_list = np.array([70.14, 72.54, 86.52, 85.26, 84.96, 80.17, 85.10, 85.87, 95.37])

# 2. 设置 Seaborn 风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# 3. 定义颜色与标记
#   - 其余模型用同一柔和色系，不同深浅
#   - Proposed-Dual_EmoNet 用单独颜色
base_color = "#80b1d3"  # 柔和蓝色
highlight_color = "#fb8072"  # 柔和红色
markers = ["o"] * (len(models) - 1) + ["D"]
model_colors = [base_color]*(len(models) - 1) + [highlight_color]

# 4. 作图：Params vs. Accuracy
plt.figure(figsize=(8, 6))
texts = []
for i, model in enumerate(models):
    x = params_list[i]
    y = accuracy_list[i]
    plt.scatter(x, y, color=model_colors[i], marker=markers[i], s=100, alpha=0.8)
    # 添加标签，但暂时不指定位置
    texts.append(plt.text(x, y, model, fontsize=10, ha='left', va='bottom'))

plt.xlabel("Parameters (M)")
plt.ylabel("Accuracy (%) on RAF-DB")
plt.title("Model Comparison (Params vs. Accuracy)")

# 自动调整标签，避免重叠
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

# 设置坐标范围，使图更紧凑；根据你的数据可适当调整
plt.xlim(0, max(params_list)*1.1)
plt.ylim(min(accuracy_list)*0.95, 100)

plt.tight_layout()
plt.savefig("comparison_params.png", dpi=300)
plt.show()

# 5. 作图：FLOPs vs. Accuracy
plt.figure(figsize=(8, 6))
texts = []
for i, model in enumerate(models):
    x = flops_list[i]
    y = accuracy_list[i]
    plt.scatter(x, y, color=model_colors[i], marker=markers[i], s=100, alpha=0.8)
    texts.append(plt.text(x, y, model, fontsize=10, ha='left', va='bottom'))

plt.xlabel("FLOPs (G)")
plt.ylabel("Accuracy (%) on RAF-DB")
plt.title("Model Comparison (FLOPs vs. Accuracy)")

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

# 同样设置坐标范围
plt.xlim(0, max(flops_list)*1.1)
plt.ylim(min(accuracy_list)*0.95, 100)

plt.tight_layout()
plt.savefig("comparison_flops.png", dpi=300)
plt.show()
