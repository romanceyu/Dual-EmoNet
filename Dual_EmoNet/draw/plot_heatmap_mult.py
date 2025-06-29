import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义情绪标签列表
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']


model_dir = '/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models'
csv_path = os.path.join(model_dir, 'classification_scores.csv')
# 读取验证集的分类得分CSV文件
df = pd.read_csv(csv_path)

# 将所有情绪列转换为数值类型
# errors='coerce' 参数会将无法转换的值设为 NaN
for emotion in emotions:
    df[emotion] = pd.to_numeric(df[emotion], errors='coerce')

# 创建空的混淆矩阵，使用pandas DataFrame
# index和columns都使用情绪标签
# 初始值全部设为0
confusion_matrix = pd.DataFrame(0, index=emotions, columns=emotions)

# 遍历数据集中的每一行
for _, row in df.iterrows():
    # 从文件路径中提取真实标签
    # 假设文件名格式为：xxx_emotion.jpg
    true_label = row['filepath'].split('_')[-1].split('.')[0]
    true_label = true_label.replace('sadness', 'sad')

    # 获取预测标签（得分最高的情绪）
    # astype(float)确保进行数值比较
    # idxmax()返回最大值的索引（这里是情绪标签）
    predicted_label = row[emotions].astype(float).idxmax()

    # 在混淆矩阵中对应位置加1
    confusion_matrix.loc[true_label, predicted_label] += 1

# 对混淆矩阵进行归一化
# 按行归一化，使每行总和为1
# 这样可以显示每个真实类别被预测为各个类别的概率
confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1)

# 创建图形，设置大小为8x6英寸
plt.figure(figsize=(8, 6))

# 使用seaborn绘制热力图
sns.heatmap(
    confusion_matrix_normalized,  # 使用归一化后的混淆矩阵
    annot=True,  # 显示数值标注
    cmap='Oranges',  # 使用橙色色图
    fmt='.2f',  # 数值格式化为保留2位小数
    xticklabels=emotions,  # X轴标签使用情绪标签
    yticklabels=emotions  # Y轴标签使用情绪标签
)

# 设置坐标轴标签
plt.xlabel('Predicted Label')  # 预测标签
plt.ylabel('True Label')  # 真实标签

# 保存图像
plt.savefig(
    'validation.png',  # 保存文件名
    dpi=400,  # 分辨率设置为400dpi
    bbox_inches='tight',  # 自动调整边界，确保标签完整显示
    pad_inches=0.1  # 设置小边距
)

# 显示图像
plt.show()