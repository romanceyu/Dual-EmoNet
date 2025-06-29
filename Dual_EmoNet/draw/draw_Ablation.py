import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 学术可视化配置 -----------------
sns.set(style='whitegrid', context='paper', font_scale=1.3)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300,
    'savefig.format': 'png',
    'svg.fonttype': 'none'
})

# ----------------- 数据准备 -----------------
# 模型编号（0-9号模型）
models = np.arange(10)

# 根据表 4-3 的数据
test_acc = np.array([86.52, 89.10, 92.54, 91.68, 90.81, 92.20, 93.41, 94.16, 93.50, 95.37])
f1_score  = np.array([0.809, 0.847, 0.901, 0.883, 0.873, 0.892, 0.908, 0.910, 0.906, 0.933])

# ----------------- 创建保存绘图的目录 -----------------
results_dir = '/root/autodl-tmp/ResEmoteNet/runs/results'
plots_dir = os.path.join(results_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# ----------------- 图 4-9: Dual_EmoNet 消融实验 Test Accuracy 折线对比图 -----------------
plt.figure(figsize=(8, 6))
# 采用柔和的 pastel 蓝色
line_color = '#a6cee3'
plt.plot(models, test_acc, marker='o', linestyle='-', color=line_color, linewidth=2, markersize=8)
for i, acc in enumerate(test_acc):
    plt.text(models[i], acc + 0.5, f"{acc:.2f}", ha='center', va='bottom', fontsize=12, color=line_color)
plt.xlabel("Model Index", fontsize=14)
plt.ylabel("Test Accuracy (%)", fontsize=14)
plt.title("Ablation Experiment - Test Accuracy", fontsize=14, fontweight='bold')
plt.xticks(models)
plt.ylim(80, 100)
plt.tight_layout()
test_acc_path = os.path.join(plots_dir, "Dual_EmoNet_Test_Acc.png")
plt.savefig(test_acc_path, dpi=300)
plt.close()

# ----------------- 图 4-10: Dual_EmoNet 消融实验 F1-Score 折线对比图 -----------------
plt.figure(figsize=(8, 6))
# 采用柔和的 pastel 橙色
line_color = '#fdbf6f'
plt.plot(models, f1_score, marker='o', linestyle='-', color=line_color, linewidth=2, markersize=8)
for i, f1 in enumerate(f1_score):
    plt.text(models[i], f1 + 0.005, f"{f1:.3f}", ha='center', va='bottom', fontsize=12, color=line_color)
plt.xlabel("Model Index", fontsize=14)
plt.ylabel("F1-Score", fontsize=14)
plt.title("Ablation Experiment - F1-Score", fontsize=14, fontweight='bold')
plt.xticks(models)
plt.ylim(0.75, 1.0)
plt.tight_layout()
f1_score_path = os.path.join(plots_dir, "Dual_EmoNet_F1_Score.png")
plt.savefig(f1_score_path, dpi=300)
plt.close()

print(f"Plots saved to:\n{test_acc_path}\n{f1_score_path}")
