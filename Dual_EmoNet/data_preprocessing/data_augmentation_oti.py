
import random
from pathlib import Path
from PIL import Image, ImageOps
from collections import defaultdict

# 配置参数
dataset_path = Path("/root/autodl-tmp/ResEmoteNet/fer_im/fer/train")
target_num = 1200  # 每个类别目标数量


def safe_augment(img):
    """安全增强策略（保持原逻辑）"""
    transforms = [
        (ImageOps.mirror, 0.3),
        (lambda x: x.rotate(random.gauss(0, 10)), 0.4),
        (lambda x: ImageOps.crop(x, border=random.randint(5, 15)), 0.3),
        (lambda x: x.resize((int(x.width * 0.9), int(x.height * 0.9))), 0.3)
    ]
    for transform, prob in transforms:
        if random.random() < prob:
            img = transform(img)
    return img


def process_category(emotion_files, emotion):
    """处理单个情感类别"""
    current_count = len(emotion_files)
    if current_count >= target_num:
        print(f"跳过 {emotion}（已有 {current_count} 张）")
        return

    # 计算需要生成的增强数量
    need_aug = target_num - current_count
    print(f"\n处理 {emotion} 类别：")
    print(f"原始数量: {current_count} | 需要增强: {need_aug}")

    # 增强计数器
    aug_count = 0
    idx = 0  # 原始文件索引

    while aug_count < need_aug:
        # 循环使用原始文件
        if idx >= current_count:
            idx = 0

        src_file = dataset_path / emotion_files[idx]
        try:
            # 读取并增强
            img = Image.open(src_file)
            aug_img = safe_augment(img)

            # 生成唯一文件名
            new_name = f"{src_file.stem}_aug{aug_count}_{emotion}{src_file.suffix}"
            new_path = dataset_path / new_name

            # 保存增强后的图片
            aug_img.save(new_path)

            aug_count += 1
            idx += 1  # 移动到下一个原始文件

            # 每完成50张打印进度
            if aug_count % 50 == 0:
                print(f"已生成 {aug_count}/{need_aug} 张增强图片")

        except Exception as e:
            print(f"增强失败 {src_file}: {str(e)}")
            idx += 1  # 失败时也移动到下一个文件

    print(f"{emotion} 类别增强完成，最终数量：{target_num}")


if __name__ == "__main__":
    # 初始化数据结构
    emotion_files = defaultdict(list)
    for f in dataset_path.glob("*.*"):
        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            emotion = f.stem.split("_")[-1]
            emotion_files[emotion].append(f.name)

    # 处理每个类别
    for emotion, files in emotion_files.items():
        process_category(files.copy(), emotion)

    # 最终统计
    print("\n最终统计:")
    for emotion in emotion_files:
        count = len(list(dataset_path.glob(f"*_{emotion}.*")))
        print(f"{emotion}: {count} 张")