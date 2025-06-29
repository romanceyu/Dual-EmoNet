import os
import shutil

# 定义源数据目录
source_dir = 'data_dir'

# 定义目标数据目录
destination_dir = 'out_dir'

# 定义文件夹名称映射关系
# 将原始文件夹名映射为标准化的短名称
folder_mapping = {
    'test': 'test',  # 测试集文件夹映射
    'train': 'train',  # 训练集文件夹映射
    'validation': 'val'  # 验证集文件夹映射
}

# 遍历主要数据集文件夹（测试集、训练集、验证集）
for folder in ['test', 'train', 'validation']:
    # 构建当前数据集文件夹的完整路径
    folder_path = os.path.join(source_dir, folder)

    # 遍历每个数据集文件夹中的类别文件夹
    for class_folder in os.listdir(folder_path):
        # 构建类别文件夹的完整路径
        class_folder_path = os.path.join(folder_path, class_folder)

        # 遍历类别文件夹中的所有图像文件
        for index, image in enumerate(os.listdir(class_folder_path)):
            # 分离图像文件名和扩展名
            image_name, image_ext = os.path.splitext(image)

            # 构建新的图像文件名
            # 格式：数据集类型_序号_表情类别.扩展名
            # 例如：train_0_happy.jpg
            new_image_name = f"{folder_mapping[folder]}_{index}_{class_folder}{image_ext}"

            # 移动并重命名图像文件到目标目录
            shutil.move(
                os.path.join(class_folder_path, image),  # 源文件路径
                os.path.join(destination_dir, folder, new_image_name)  # 目标文件路径
            )