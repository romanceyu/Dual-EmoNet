import os
import shutil
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


def get_dataset_stats(directory):
    """获取数据集统计信息"""
    total_files = 0
    valid_extensions = ('.png', '.jpg', '.jpeg')

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                total_files += 1
    return total_files


def find_similar_images_and_filter(train_dir, test_dir, val_dir, output_dir, threshold=5, train_remove_ratio=0.50, val_remove_ratio=0.5):
    """
    1. 计算 `train`, `test`, `val` 数据集的哈希值
    2. 识别 `train` 和 `test` 的相似图片，并 90% 概率删除 `train` 里的重复图片
    3. 识别 `val` 和 `train` 的相似图片，并 50% 概率删除 `val` 里的重复图片
    4. 复制处理后的数据集到 `output_dir`
    """

    # 计算哈希
    print("\n计算训练集哈希...")
    train_hashes = defaultdict(list)
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                h = compute_phash(path)
                if h:
                    train_hashes[h].append(path)

    print("计算测试集哈希...")
    test_hashes = defaultdict(list)
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                h = compute_phash(path)
                if h:
                    test_hashes[h].append(path)

    print("计算验证集哈希...")
    val_hashes = defaultdict(list)
    for root, _, files in os.walk(val_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                h = compute_phash(path)
                if h:
                    val_hashes[h].append(path)

    # 记录删除的图片数量
    deleted_train_count = 0
    deleted_val_count = 0

    # 找到 `train` 和 `test` 之间的重复项，删除train中的图片
    duplicate_train_images = set()
    for tr_hash, tr_files in train_hashes.items():
        for t_hash, t_files in test_hashes.items():
            if tr_hash - t_hash <= threshold:
                for tr_path in tr_files:
                    if random.random() < train_remove_ratio:
                        duplicate_train_images.add(tr_path)
                        deleted_train_count += 1

    # 找到 `val` 和 `train` 之间的重复项，删除val中的图片
    duplicate_val_images = set()
    for v_hash, v_files in val_hashes.items():
        for tr_hash, tr_files in train_hashes.items():
            if v_hash - tr_hash <= threshold:
                for v_path in v_files:
                    if random.random() < val_remove_ratio:
                        duplicate_val_images.add(v_path)
                        deleted_val_count += 1

    print(f"\n找到的 `train` 重复图片数量: {len(duplicate_train_images)}")
    print(f"找到的 `val` 重复图片数量: {len(duplicate_val_images)}")
    print(f"\n实际删除 `train` 图片数量: {deleted_train_count}")
    print(f"实际删除 `val` 图片数量: {deleted_val_count}")

    # 复制数据集到新的目录
    for dataset_name, dataset_dir, duplicate_set in [
        ("train", train_dir, duplicate_train_images),  # 删除train中的重复
        ("test", test_dir, set()),                     # test保留全部
        ("val", val_dir, duplicate_val_images),        # 删除val中的重复
    ]:
        new_dataset_path = os.path.join(output_dir, dataset_name)
        os.makedirs(new_dataset_path, exist_ok=True)

        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(new_dataset_path, os.path.relpath(old_path, dataset_dir))

                    # 确保目标目录存在
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)

                    # 如果文件不在删除列表中，则复制
                    if old_path not in duplicate_set:
                        shutil.copy2(old_path, new_path)

    print(f"\n处理完成！新的数据集已保存在: {output_dir}")
    print(f"最终保留的 `train` 图片数: {get_dataset_stats(os.path.join(output_dir, 'train'))}")
    print(f"最终保留的 `test` 图片数: {get_dataset_stats(os.path.join(output_dir, 'test'))}")
    print(f"最终保留的 `val` 图片数: {get_dataset_stats(os.path.join(output_dir, 'val'))}")


if __name__ == "__main__":
    # 原始数据集路径
    train_dir = "/root/autodl-tmp/ResEmoteNet/rafdb/train"
    test_dir = "/root/autodl-tmp/ResEmoteNet/rafdb/test"
    val_dir = "/root/autodl-tmp/ResEmoteNet/rafdb/val"

    # 输出数据集路径
    output_dir = "/root/autodl-tmp/ResEmoteNet/rafdb_tr"

    # 运行去重并重新划分数据集
    find_similar_images_and_filter(
        train_dir, test_dir, val_dir, output_dir,
        train_remove_ratio=0.90,  # 90%概率删除train中的重复
        val_remove_ratio=0.9      # 50%概率删除val中的重复（保持原逻辑）
    )