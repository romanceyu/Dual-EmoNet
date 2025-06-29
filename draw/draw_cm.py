import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# 设置字体
matplotlib.rc('font', family='DejaVu Sans')

# ------------------- 1. 读取数据 -------------------
#model_dir = '/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models'
#csv_path = os.path.join(model_dir, '89.1_classification_scores.csv')
#df = pd.read_csv(csv_path)
model_dir = '/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models'
csv_path = os.path.join(model_dir, '93.5_classification_scores.csv')
df = pd.read_csv(csv_path)

# 定义情绪类别（按照图片中的顺序排列）
emotions = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']  # 注意这里有重复，请根据实际情况调整
emotions_correct = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'anger', 'neutral']  # 与图片一致的顺序
labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']  # 首字母大写的标签



# 将所有情绪列转换为数值类型
for emotion in emotions_correct:
    df[emotion] = pd.to_numeric(df[emotion], errors='coerce')

# ------------------- 2. 构建混淆矩阵 -------------------
confusion_matrix = pd.DataFrame(0, index=emotions_correct, columns=emotions_correct)

# 遍历数据集中的每一行
for _, row in df.iterrows():
    # 从文件路径中提取真实标签
    true_label = row['filepath'].split('_')[-1].split('.')[0]
    # 如果文件名里可能出现 "sadness" 等字样，统一修正
    true_label = true_label.replace('sadness', 'sad')

    # 获取预测标签：选取概率最大的那一列
    predicted_label = row[emotions].astype(float).idxmax()

    # 在混淆矩阵中累加计数
    if true_label in confusion_matrix.index and predicted_label in confusion_matrix.columns:
        confusion_matrix.loc[true_label, predicted_label] += 1

# 归一化混淆矩阵
confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0) * 100  # 转换为百分比

# 重新排序行和列以匹配图片顺序
confusion_matrix_normalized = confusion_matrix_normalized.loc[emotions_correct, emotions_correct]

# ------------------- 3. 绘制热力图 -------------------
plt.figure(figsize=(10, 8))
# 创建热力图
ax = sns.heatmap(
    confusion_matrix_normalized,
    annot=True,  # 在格子中显示数值
    cmap='Blues',  # 蓝色色图
    fmt='.1f',  # 显示一位小数
    xticklabels=labels,  # X轴标签
    yticklabels=labels,  # Y轴标签
    annot_kws={'size': 14, 'weight': 'bold'},  # 数字加粗
    linewidths=0.5,  # 单元格间隔线
    vmin=0.0,  # 颜色范围从0开始
    vmax=100.0,  # 百分比最大值100
    square=True,  # 保持单元格为正方形
    cbar_kws={'label': '%', 'shrink': 0.8}  # 颜色条设置
)

# 添加边框
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(1.5)

# 标题和坐标轴设置
plt.title('RAF-DB', fontsize=18, fontweight='bold', y=1.05)  # 与图片一致

# 调整轴标签位置和样式
plt.xlabel('Predicted', fontsize=16, fontweight='bold', labelpad=15)
plt.ylabel('True', fontsize=16, fontweight='bold', labelpad=15)

# 调整刻度标签大小和位置
plt.xticks(fontsize=14, rotation=0)  # x轴标签不旋转
plt.yticks(fontsize=14, rotation=90)  # y轴标签旋转

# 在每个单元格文本后添加%符号
for text in ax.texts:
    current_text = text.get_text()
    if current_text != '':
        text.set_text(f"{float(current_text):.1f}%")

# 调整边距
plt.tight_layout()

# ------------------- 4. 保存图表 -------------------
cm_dir = '/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models/test_matrix'
if not os.path.exists(cm_dir):
    os.makedirs(cm_dir)

save_path = os.path.join(cm_dir, 'Confusion_Matrix.pdf')
plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(cm_dir, 'Confusion_Matrix.png'), dpi=800, bbox_inches='tight', pad_inches=0.1)  # 同时保存PNG版本方便预览
plt.show()