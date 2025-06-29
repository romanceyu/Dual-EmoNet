import os
import sys
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


def find_similar_images(train_dir, test_dir, threshold=5):
    # 获取数据集统计信息
    train_total = get_dataset_stats(train_dir)
    test_total = get_dataset_stats(test_dir)
    print(f"\n数据集统计:")
    print(f"训练集总图片数 ({train_dir}): {train_total}")  # 修改1：显示实际路径
    print(f"测试集总图片数 ({test_dir}): {test_total}")  # 修改1：显示实际路径

    # 计算哈希
    print("\n计算训练集哈希...")
    train_hashes = defaultdict(list)
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                h = compute_phash(path)
                if h: train_hashes[h].append(path)

    print("计算测试集哈希...")
    test_hashes = defaultdict(list)
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                h = compute_phash(path)
                if h: test_hashes[h].append(path)

    # 查找相似项
    duplicates = []
    print("\n检测相似图片...")
    for t_hash, t_files in test_hashes.items():
        for tr_hash, tr_files in train_hashes.items():
            if t_hash - tr_hash <= threshold:
                for t_path in t_files:
                    for tr_path in tr_files:
                        duplicates.append((tr_path, t_path))

    # 保存结果到父目录
    parent_dir = os.path.dirname(os.path.normpath(train_dir))
    output_path = os.path.join(parent_dir, "duplicates_report.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==== 数据集相似性检测报告 ====\n")
        f.write(f"训练集路径: {train_dir}\n")
        f.write(f"测试集路径: {test_dir}\n")
        f.write(f"训练集图片总数: {train_total}\n")
        f.write(f"测试集图片总数: {test_total}\n")
        f.write(f"\n发现相似图片对（阈值={threshold}）: {len(duplicates)}\n")

        if duplicates:
            f.write("\n匹配详情:\n")
            for idx, (train_p, test_p) in enumerate(duplicates, 1):
                f.write(f"{idx}. 训练集: {train_p}\n   测试集: {test_p}\n")
        else:
            f.write("\n未发现相似图片")

    # 修改2：添加结果打印语言
    print(f"\n找到的相似图片对数量: {len(duplicates)}")
    print(f"\n检测完成！报告已保存至: {output_path}")

if __name__ == "__main__":
    # 使用原始字符串处理Windows路径
    train_dir = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/train"
    test_dir = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/test"
    find_similar_images(train_dir, test_dir)