import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# ----------------- 学术可视化配置 -----------------
sns.set(style='whitegrid', context='paper', font_scale=1.3)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300,
    'savefig.format': 'png',
    'svg.fonttype': 'none'
})

# ----------------- 数据准备 -----------------
# x轴类别：情绪顺序为：['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']
n_emotions = len(emotions)

# 根据要求重新排列后的数据（顺序：[surprise, fear, disgust, happy, sad, angry, neutral]）
data = {
    "POSTER":       [90.27, 67.57, 75.00, 96.96, 91.21, 88.89, 92.35],
    "POSTER++":     [90.58, 68.92, 71.88, 97.22, 92.89, 88.27, 92.06],
    "APViT":        [93.00, 73.00, 74.00, 97.00, 89.00, 86.00, 92.00],
    "QCS":          [91.20, 68.90, 79.40, 97.10, 91.80, 88.30, 92.20],
    "Dual_EmoNet":  [92.10, 83.80, 75.00, 96.90, 93.30, 91.40, 92.90]
}

# 设置每个方法的柔和 pastel 色彩
colors = {
    "POSTER": "#a6cee3",
    "POSTER++": "#b2df8a",
    "APViT": "#fb9a99",
    "QCS": "#fdbf6f",
    "Dual_EmoNet": "#cab2d6"
}

methods = list(data.keys())
n_methods = len(methods)

# 为避免同一情绪下的散点重叠，给每个方法设置微小的x轴偏移
offsets = np.linspace(-0.2, 0.2, n_methods)

# ----------------- 创建保存绘图的目录 -----------------
results_dir = '/root/autodl-tmp/ResEmoteNet/runs/results'
plots_dir = os.path.join(results_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# ----------------- 绘制散点图 -----------------
plt.figure(figsize=(10, 7))
all_texts = []  # 用于存储所有文本对象以调整位置

for i, method in enumerate(methods):
    scores = data[method]
    # 计算带偏移的x坐标：原始x坐标为0,1,...,n_emotions-1，再加上对应偏移
    x_coords = np.arange(n_emotions) + offsets[i]
    # 绘制散点，点的大小增大为150
    plt.scatter(x_coords, scores, s=150, color=colors[method], label=method, zorder=3)
    # 添加数据标注，初始位置设置在对应数据点上方 0.5 单位
    for x, y in zip(x_coords, scores):
        txt = plt.text(x, y + 0.5, f"{y:.1f}", ha='center', va='bottom',
                       fontsize=10, color=colors[method], zorder=4)
        all_texts.append(txt)

# 设置x轴刻度仅显示实际存在的情绪标签，使用整数位置对应
plt.xticks(np.arange(n_emotions), emotions, fontsize=12)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.title("SOTA Model Emotion Classification Accuracy", fontsize=14, fontweight='bold')

# 根据数据范围自动调整y轴刻度，仅显示实际存在的区间
all_scores = np.concatenate(list(data.values()))
y_min, y_max = np.floor(all_scores.min()/5)*5, np.ceil(all_scores.max()/5)*5
plt.ylim(y_min, y_max)

# 调整标注位置，避免重叠
adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

# 设置图例放置在右下角（该区域无数据点），并稍微内嵌到图中
plt.legend(title="Method", loc="lower right", fontsize=12)
plt.tight_layout()

# 保存图像（dpi=300）
save_path = os.path.join(plots_dir, "Dual_EmoNet_Emotion_Accuracy.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved to: {save_path}")
