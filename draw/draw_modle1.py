import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# 1. 准备数据（Params vs. Accuracy，单位：M & %）
models_params = [
    "RAN", "VTFF", "SCAN-CCI", "DMUE", "EAC", "ARM",
    "TransFER", "Facial Chirality", "DDAMFN", "APVIT", "POSTER", "POSTER++", "Dual_EmoNet"
]
params_list = np.array([11.2, 80.1, 70, 78.4, 11.2, 11.2, 65.2, 46.2, 4.11, 65.2, 71.8, 43.7, 42.49])
accuracy_list = np.array([86.90, 88.14, 89.02, 89.42, 89.99, 90.42, 90.91, 91.20, 91.35, 91.98, 92.05, 92.21, 95.37])

# 2. 设置 Seaborn 风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


# 3. 定义颜色与标记
# 对于所有模型采用相同的柔和蓝色，Dual_EmoNet 用单独的柔和红色突出显示
base_color = "#80b1d3"
highlight_color = "#fb8072"
# 这里假设最后一个模型 Dual_EmoNet 为高亮
markers = ["o"] * (len(models_params) - 1) + ["D"]
model_colors = [base_color]*(len(models_params) - 1) + [highlight_color]

# 4. 作图：Params vs. Accuracy
plt.figure(figsize=(8, 6))
texts = []
for i, model in enumerate(models_params):
    x = params_list[i]
    y = accuracy_list[i]
    plt.scatter(x, y, color=model_colors[i], marker=markers[i], s=100, alpha=0.8)
    texts.append(plt.text(x, y, model, fontsize=10, ha='left', va='bottom'))

plt.xlabel("Parameters (M)")
plt.ylabel("Accuracy (%) on RAF-DB")
plt.title("Model Comparison (Params vs. Accuracy)")
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
plt.xlim(0, params_list.max()*1.1)
plt.ylim(accuracy_list.min()*0.95, 100)
plt.tight_layout()
plt.savefig("comparison_params.png", dpi=300)
plt.show()

# 5. 准备 FLOPs 数据（仅针对有 FLOPs 数据的模型，单位：G）
models_flops = ["TransFER", "APVIT", "POSTER", "POSTER++", "DMUE", "Dual_EmoNet"]
flops_list = np.array([15.3, 12.67, 15.7, 8.4, 13.4, 0.46])
# 从上面的 RAF-DB 数据中提取对应模型的 Accuracy
# 对应顺序：TransFER 90.91, APVIT 91.98, POSTER 92.05, POSTER++ 92.21, DMUE 89.42, Dual_EmoNet 95.37
accuracy_flops = np.array([90.91, 91.98, 92.05, 92.21, 89.42, 95.37])

# 6. 作图：FLOPs vs. Accuracy
plt.figure(figsize=(8, 6))
texts = []
for i, model in enumerate(models_flops):
    x = flops_list[i]
    y = accuracy_flops[i]
    plt.scatter(x, y, color=model_colors[-(len(models_flops)-i)], marker=markers[-(len(models_flops)-i)], s=100, alpha=0.8)
    texts.append(plt.text(x, y, model, fontsize=10, ha='left', va='bottom'))

plt.xlabel("FLOPs (G)")
plt.ylabel("Accuracy (%) on RAF-DB")
plt.title("Model Comparison (FLOPs vs. Accuracy)")
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
plt.xlim(0, flops_list.max()*1.1)
plt.ylim(accuracy_flops.min()*0.95, 100)
plt.tight_layout()
plt.savefig("comparison_flops.png", dpi=300)
plt.show()
