# 导入必要的库
import os  # 操作系统接口，用于处理文件路径
import torch  # PyTorch深度学习框架
import pandas as pd  # 数据处理库，用于读取CSV标签文件
from PIL import Image  # 图像处理库，用于加载图像文件
from torch.utils.data import Dataset  # PyTorch数据集基类


# 自定义数据集类，继承自PyTorch的Dataset
class Four4All(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        构造函数，初始化数据集
        :param csv_file:  CSV文件路径，包含图像文件名和对应标签（如面部表情类别）
        :param img_dir:   图像文件夹根目录（如 '/data/expressions/'）
        :param transform: 图像预处理和数据增强操作（如转为Tensor、归一化、随机裁剪等）
        """
        # 读取CSV文件，假设格式为两列：[图像文件名, 表情标签编号]
        # 示例CSV内容：
        # image_name,label
        # face_001.jpg,0
        # face_002.jpg,3
        # 标签编号对应具体表情（如0=生气，1=厌恶，2=恐惧，3=开心等）
        self.labels = pd.read_csv(csv_file)

        # 存储图像根目录（如 '/data/images/'）
        self.img_dir = img_dir

        # 存储图像预处理和数据增强操作（如Compose组合多个变换）
        self.transform = transform

    def __len__(self):
        """返回数据集样本总数（必须实现的方法）"""
        return len(self.labels)  # 直接返回CSV行数

    def __getitem__(self, idx):
        """
        获取单个样本（必须实现的方法）
        :param idx: 样本索引（支持整数或张量索引）
        :return: 预处理后的图像张量，对应的表情标签
        """
        # 如果索引是张量（如多GPU训练时），转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 拼接完整的图像文件路径（假设CSV第一列为文件名）
        # 例如：self.img_dir = '/data/images/', CSV中文件名为 'face_001.jpg'
        # 最终路径 -> '/data/images/face_001.jpg'
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])

        # 使用PIL加载图像（面部表情图像通常为RGB或灰度图）
        image = Image.open(img_name)  # 返回PIL.Image对象

        # 获取标签（假设CSV第二列为表情标签）
        label = self.labels.iloc[idx, 1]

        # 应用预处理和数据增强（关键步骤！）
        if self.transform:
            image = self.transform(image)  # 例如：转为Tensor并归一化到[-1,1]

        # 返回图像张量和标签（标签自动转为PyTorch张量）
        return image, label