import os
import shutil  # 用于文件操作的系统工具库


def move_images(source_folder, destination_folder):
    """
    递归遍历源文件夹，移动所有图片文件到目标文件夹

    Args:
        source_folder: 源文件夹路径，包含需要移动的图片
        destination_folder: 目标文件夹路径，图片将被移动到这里
    """
    # os.walk 递归遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 只处理 jpg 和 png 格式的图片文件
            if file.endswith('.jpg') or file.endswith('.png'):
                # 构建完整的源文件路径
                source_path = os.path.join(root, file)
                # 构建完整的目标文件路径
                destination_path = os.path.join(destination_folder, file)
                # 移动文件
                shutil.move(source_path, destination_path)


# 源数据文件夹路径
test_folder = 'data_dir'

# 目标文件夹路径
destination_folder = 'out_dir'

# 执行文件移动操作
move_images(test_folder, destination_folder)