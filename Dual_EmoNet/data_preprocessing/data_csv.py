import os
import pandas as pd
from pathlib import Path

# 配置参数
data_dir = '/root/autodl-tmp/ResEmoteNet/fer_im/fer/train'
output_dir = str(Path(data_dir).parent)  # 获取与val同级的父目录
label_mapping = {
    "happy": 0,
    "surprise": 1,
    "sad": 2,
    "angry": 3,
    "disgust": 4,
    "fear": 5,
    "neutral": 6
}


def generate_label_csv():
    # 创建存储容器
    image_data = []

    # 遍历验证集目录
    for filename in os.listdir(data_dir):
        # 文件过滤和标签解析
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 增强文件名解析逻辑
            parts = filename.split('_')
            emotion = parts[-1].split('.')[0]  # 提取最后一段作为情绪标签

            # 标签有效性验证
            if emotion in label_mapping:
                label_value = label_mapping[emotion]
                image_data.append([filename, label_value])
            else:
                print(f"警告：发现未知标签 '{emotion}' 在文件 {filename}")

    # 创建DataFrame并保存
    df = pd.DataFrame(image_data, columns=["filename", "label"])
    csv_path = os.path.join(output_dir, "train_labels.csv")

    # 保存文件（包含列名）
    df.to_csv(csv_path, index=False)
    print(f"CSV文件已生成于：{csv_path}")
    print(f"共处理 {len(df)} 个有效样本")


if __name__ == "__main__":
    generate_label_csv()