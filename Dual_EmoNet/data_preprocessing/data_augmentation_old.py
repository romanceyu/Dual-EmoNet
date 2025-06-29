import os
from PIL import Image, ImageOps  # 导入PIL库用于图像处理
import random

# 定义数据集路径和每个类别期望的图像数量
dataset_path = '/root/autodl-tmp/ResEmoteNet/rafdb_basic_after_clustering/train'  # 数据集所在的目录路径
target_num_images = 1000  # 数据增强后每个表情类别期望达到的图像数量


def augment_image(image):
    """
    对输入图像应用随机数据增强操作

    Args:
        image: PIL.Image对象，输入的原始图像

    Returns:
        PIL.Image对象，经过随机增强后的图像
    """
    # 定义可用的增强操作列表
    augmentations = [
        lambda x: x.rotate(random.randint(-30, 30)),  # 随机旋转，角度范围[-30,30]度
        lambda x: ImageOps.mirror(x),  # 水平翻转
        lambda x: ImageOps.crop(x, border=random.randint(0, 10)),  # 随机裁剪，边界范围[0,10]像素
    ]
    # 随机选择一种增强操作
    augmentation = random.choice(augmentations)
    return augmentation(image)



# 按表情类别对图像进行分组
images_by_emotion = {}  # 创建字典存储每个表情类别的图像列表
for filename in os.listdir(dataset_path):  # 遍历数据集目录
    if filename.endswith(('.jpg', '.png')):  # 只处理jpg和png格式的图像
        # 从文件名中提取表情类别标签
        emotion = filename.split('_')[-1].split('.')[0]
        # 将图像按表情类别分组
        if emotion not in images_by_emotion:
            images_by_emotion[emotion] = []
        images_by_emotion[emotion].append(filename)

# 对每个表情类别进行数据增强
for emotion, images in images_by_emotion.items():
    num_images = len(images)  # 获取当前类别的原始图像数量

    # 如果当前类别的图像数量少于目标数量，进行数据增强
    if num_images < target_num_images:
        # 循环生成新的增强图像，直到达到目标数量
        for i in range(target_num_images - num_images):
            # 使用取模操作循环使用原始图像
            original_image_path = os.path.join(dataset_path, images[i % num_images])
            print(original_image_path)
            # 打开原始图像并进行增强
            with Image.open(original_image_path) as img:
                augmented_image = augment_image(img)
                # 生成新图像的文件名并保存
                new_image_name = f'img_{num_images + i + 1}_{emotion}.jpg'
                augmented_image.save(os.path.join(dataset_path, new_image_name))

print("Image augmentation completed.")  # 打印完成提示
