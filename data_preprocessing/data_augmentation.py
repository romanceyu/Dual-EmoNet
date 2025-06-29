import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageOps
from collections import defaultdict

# 配置参数
dataset_path = Path("/root/autodl-tmp/ResEmoteNet/rafdb_basic_after_clustering/train")
val_path = Path("/root/autodl-tmp/ResEmoteNet/rafdb_basic_after_clustering/val")
target_num = 1000
val_ratio = 0.2


def safe_augment(img):
    """安全增强策略"""
    transforms = [
        (ImageOps.mirror, 0.3),  # 30% 概率进行水平翻转
        (lambda x: x.rotate(random.gauss(0, 10)), 0.4),  # 旋转角度服从 N(0,10) 的高斯分布
        (lambda x: ImageOps.crop(x, border=random.randint(5, 15)), 0.3),  # 随机裁剪边缘
        (lambda x: x.resize((int(x.width*0.9), int(x.height*0.9))), 0.3)  # 随机缩小图片尺寸
    ]
    for transform, prob in transforms:
        if random.random() < prob:
            img = transform(img)
    return img

def process_category(emotion_files, emotion):
        # === 新增目录创建代码 ===
    val_path.mkdir(parents=True, exist_ok=True)
    # 创建隔离环境
    temp_dir = dataset_path.parent / "temp_processing"
    temp_dir.mkdir(exist_ok=True)

    # 预分割原始数据
    random.shuffle(emotion_files)
    split_idx = int(len(emotion_files) * (1 - val_ratio))
    origin_train = emotion_files[:split_idx]
    origin_val = emotion_files[split_idx:]

    # === 阶段1：处理训练部分 ===
    # 计算需要生成的增强数量
    need_aug = target_num - len(origin_train)

    # 移动原始训练文件到临时目录
    train_pool = []
    for fname in origin_train:
        src = dataset_path / fname
        dest = temp_dir / fname
        shutil.move(str(src), str(dest))
        train_pool.append(dest)

    # 执行安全增强
    aug_count = 0
    while len(train_pool) < target_num:
        src_file = train_pool[aug_count % len(origin_train)]
        try:
            img = Image.open(src_file)
            aug_img = safe_augment(img)

            new_name = f"{src_file.stem}_safeaug{aug_count}_{emotion}{src_file.suffix}"
            new_path = temp_dir / new_name
            aug_img.save(new_path)
            train_pool.append(new_path)
            aug_count += 1
        except Exception as e:
            print(f"增强失败 {src_file}: {str(e)}")

    # === 阶段2：处理验证部分 ===
    # 移动验证原始文件到最终验证目录
    for fname in origin_val:
        src = dataset_path / fname
        dest = val_path / fname
        shutil.move(str(src), str(dest))

    # === 阶段3：合并数据 ===
    # 将临时目录文件移回训练集
    for f in temp_dir.iterdir():
        shutil.move(str(f), str(dataset_path / f.name))
    temp_dir.rmdir()


if __name__ == "__main__":
    # 初始化数据结构
    emotion_files = defaultdict(list)
    for f in dataset_path.glob("*.*"):
        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            emotion = f.stem.split("_")[-1]
            emotion_files[emotion].append(f.name)

    # 处理每个类别
    for emotion, files in emotion_files.items():
        print(f"\n处理 {emotion} 类别...")
        process_category(files.copy(), emotion)

    # 最终统计
    train_count = {e: len(list(dataset_path.glob(f"*_{e}.*"))) for e in emotion_files}
    val_count = {e: len(list(val_path.glob(f"*_{e}.*"))) for e in emotion_files}

    print("\n最终统计:")
    for e in emotion_files:
        print(f"{e}: 训练集 {train_count[e]} | 验证集 {val_count[e]}")