import os
import random
import imagehash
from PIL import Image
from collections import defaultdict


def compute_phash(image_path):
    """计算图像的感知哈希值"""
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        print(f"处理图像错误 {image_path}: {e}")
        return None


class DatasetProcessor:
    def __init__(self, directory):
        self.dir = directory
        self.hashes = defaultdict(list)  # 哈希值到路径列表的映射
        self.path_map = {}  # 路径到哈希值的映射
        self._compute_hashes()

    def _compute_hashes(self):
        """计算数据集哈希值"""
        for root, _, files in os.walk(self.dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    h = compute_phash(path)
                    if h:
                        self.hashes[h].append(path)
                        self.path_map[path] = h

    def find_cross_duplicates(self, other, threshold):
        """查找与其他数据集的交叉重复"""
        duplicates = []
        for h in self.hashes:
            # 查找相似哈希
            similar_hashes = [oh for oh in other.hashes if h - oh <= threshold]
            for similar_h in similar_hashes:
                duplicates.extend(self.hashes[h])
        return list(set(duplicates))

    def remove_duplicates(self, duplicates, max_count):
        """
        删除重复项并更新哈希表
        :param duplicates: 需要处理的重复路径列表
        :param max_count: 允许保留的最大数量
        :return: 实际删除数量
        """
        if len(duplicates) <= max_count:
            return 0

        # 随机选择要删除的项
        remove_count = len(duplicates) - max_count
        to_remove = random.sample(duplicates, remove_count)

        # 执行删除操作
        deleted = 0
        for path in to_remove:
            if os.path.exists(path):
                os.remove(path)
                # 更新哈希表
                h = self.path_map.get(path)
                if h and path in self.hashes.get(h, []):
                    self.hashes[h].remove(path)
                    if not self.hashes[h]:
                        del self.hashes[h]
                del self.path_map[path]
                deleted += 1
        return deleted


def process_cross_datasets(main_processor, ref_processor, threshold, p):
    """处理跨数据集重复"""
    # 查找主数据集中的重复项
    duplicates = main_processor.find_cross_duplicates(ref_processor, threshold)
    print(f"\n发现 {len(duplicates)} 个 {main_processor.dir} 与 {ref_processor.dir} 的重复项")

    # 执行删除操作
    deleted = main_processor.remove_duplicates(duplicates, p)
    print(f"已删除 {deleted} 个重复项，保留 {max(len(duplicates) - deleted, 0)} 个")
    return deleted


if __name__ == "__main__":
    # 数据集路径配置
    base_dir = "/root/autodl-tmp/ResEmoteNet/fer_im/fer"
    dir_config = {
        "train": os.path.join(base_dir, "train"),
        "test": os.path.join(base_dir, "test"),
        "val": os.path.join(base_dir, "val")
    }

    # 参数配置
    hash_threshold = 5  # 哈希差异阈值
    p = 650  # 单个数据集允许的最大重复数

    # 初始化处理器（按处理优先级排序）
    processors = {
        "train": DatasetProcessor(dir_config["train"]),# 最高优先级处理
        "val": DatasetProcessor(dir_config["val"]),

        "test": DatasetProcessor(dir_config["test"])  # 最低优先级
    }

    total_deleted = 0

    # 处理流程（按优先级顺序）
    # 1. 验证集 vs 测试集
    total_deleted += process_cross_datasets(
        main_processor=processors["val"],
        ref_processor=processors["test"],
        threshold=hash_threshold,
        p=p
    )

    # 2. 验证集 vs 训练集
    total_deleted += process_cross_datasets(
        main_processor=processors["train"],
        ref_processor=processors["val"],
        threshold=hash_threshold,
        p=p
    )

    # 3. 测试集 vs 训练集
    total_deleted += process_cross_datasets(
        main_processor=processors["train"],
        ref_processor=processors["test"],
        threshold=hash_threshold,
        p=p
    )

    # 4. 测试集 vs 验证集（处理剩余重复）
    total_deleted += process_cross_datasets(
        main_processor=processors["val"],
        ref_processor=processors["test"],
        threshold=hash_threshold,
        p=p
    )

    print(f"\n总计删除 {total_deleted} 个跨数据集重复项")