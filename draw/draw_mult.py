import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置matplotlib字体
plt.rc('font', family='DejaVu Sans')

# 读取CSV文件
csv_file = '/root/autodl-tmp/ResEmoteNet/runs/results/4training_metrics.csv'  # 请替换为实际的CSV文件路径
data = pd.read_csv(csv_file)

# 获取所有模型名称
models = data['Model'].unique()

# 定义颜色和标记

# -------------------- 绘制对比图 --------------------
# 定义各模型的颜色与 marker
model_colors = {
    "baseline_res18_no_aug": "#a6cee3",
    "baseline_res18_aug": "#b2df8a",
    "baseline_arc": "#fb9a99",
    "baseline_Ghost": "#8dd3c7",       # 新增：为baseline_Ghost设置颜色
    "baseline_Pyram": "#ffffb3",       # 新增：为baseline_Pyram设置颜色
    "baseline_Dual_ResBlock": "#bebada",# 新增：为baseline_Dual_ResBlock设置颜色
    "baseline_arc_Ghost": "#cab2d6",
    "baseline_arc_Ghost_pyram": "#fdbf6f",
    "Dual_EmoNet_no_aug": "#ffff99",
    "Dual_EmoNet_full": "#1f78b4"
}

model_markers = {
    "baseline_res18_no_aug": "o",
    "baseline_res18_aug": "s",
    "baseline_arc": "^",
    "baseline_Ghost": "e",            # 新增：为baseline_Ghost设置标记
    "baseline_Pyram": "t",            # 新增：为baseline_Pyram设置标记
    "baseline_Dual_ResBlock": "*",     # 新增：为baseline_Dual_ResBlock设置标记
    "baseline_arc_Ghost": "d",
    "baseline_arc_Ghost_pyram": "v",
    "Dual_EmoNet_no_aug": "p",
    "Dual_EmoNet_full": "X"
}


# 创建保存绘图的目录
results_dir = '/root/autodl-tmp/ResEmoteNet/runs/results'
if not os.path.exists(os.path.join(results_dir, 'plots')):
    os.makedirs(os.path.join(results_dir, 'plots'))

# 收集所有模型的早停轮数
early_stop_epochs = {}
for model in models:
    model_data = data[data['Model'] == model]
    max_epoch = model_data['Epoch'].max()
    if max_epoch < 90:  # 如果没有跑满90轮，说明触发了早停
        early_stop_epochs[model] = max_epoch


# 绘制测试损失对比图
plt.figure(figsize=(10, 6))
for model in models:
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']
    test_loss = model_data['Test_Loss']
    plt.plot(epochs, test_loss, label=model, color=model_colors[model],
             marker=model_markers[model], linestyle='-', linewidth=2, markersize=6)
    # 标记早停点
    if model in early_stop_epochs:
        es_epoch = early_stop_epochs[model]
        plt.axvline(x=es_epoch, color=model_colors[model], linestyle='--', alpha=0.7)
        plt.text(es_epoch, max(test_loss)*0.9, f"Stop\n @ {es_epoch}",
                 fontsize=10, color=model_colors[model], ha='center', va='top', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
# 在x轴上标记早停轮数
if early_stop_epochs:
    plt.xticks(list(plt.xticks()[0]) + list(early_stop_epochs.values()))

current_ticks = plt.xticks()[0]
if 90 not in current_ticks:
    ticks = list(current_ticks) + [90]
    ticks.sort()
    plt.xticks(ticks)
plt.xlim(0, 92)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'plots', 'test_loss_comparison.png'), dpi=400)
plt.show()

# 绘制测试准确率对比图
plt.figure(figsize=(10, 6))
for model in models:
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']
    test_accuracy = model_data['Test_Accuracy']
    plt.plot(epochs, test_accuracy, label=model, color=model_colors[model],
             marker=model_markers[model], linestyle='-', linewidth=2, markersize=6)
    # 标记早停点
    if model in early_stop_epochs:
        es_epoch = early_stop_epochs[model]
        plt.axvline(x=es_epoch, color=model_colors[model], linestyle='--', alpha=0.7)
        plt.text(es_epoch, min(test_accuracy) * 1.1, f"Stop \n @{es_epoch}",
                 fontsize=10, color=model_colors[model], ha='center', va='bottom', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
# 在x轴上标记早停轮数
if early_stop_epochs:
    plt.xticks(list(plt.xticks()[0]) + list(early_stop_epochs.values()))

current_ticks = plt.xticks()[0]
if 90 not in current_ticks:
    ticks = list(current_ticks) + [90]
    ticks.sort()
    plt.xticks(ticks)
plt.xlim(0, 92)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'plots', 'test_accuracy_comparison.png'), dpi=400)
plt.show()

# 绘制测试F1分数对比图
plt.figure(figsize=(10, 6))
for model in models:
    model_data = data[data['Model'] == model]
    epochs = model_data['Epoch']
    test_f1 = model_data['Test_F1_Score']
    plt.plot(epochs, test_f1, label=model, color=model_colors[model],
             marker=model_markers[model], linestyle='-', linewidth=2, markersize=6)
    # 标记早停点
    if model in early_stop_epochs:
        es_epoch = early_stop_epochs[model]
        plt.axvline(x=es_epoch, color=model_colors[model], linestyle='--', alpha=0.7)
        plt.text(es_epoch, max(test_f1)*0.9, f"Stop \n @{es_epoch}",
                 fontsize=10, color=model_colors[model], ha='center', va='top', fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Test F1 Score")
plt.title("Test F1 Score Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
# 在x轴上标记早停轮数
if early_stop_epochs:
    plt.xticks(list(plt.xticks()[0]) + list(early_stop_epochs.values()))

current_ticks = plt.xticks()[0]
if 90 not in current_ticks:
    ticks = list(current_ticks) + [90]
    ticks.sort()
    plt.xticks(ticks)
plt.xlim(0, 92)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'plots', 'test_f1_comparison.png'), dpi=400)
plt.show()

# 绘制PR曲线对比图（散点图）
plt.figure(figsize=(10, 6))
for model in models:
    model_data = data[data['Model'] == model]
    precision = model_data['Test_Precision']
    recall = model_data['Test_Recall']
    plt.scatter(recall, precision, label=model, color=model_colors[model],
                marker=model_markers[model], s=30)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Scatter Plot Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'plots', 'pr_scatter_comparison.png'), dpi=400)
plt.show()