import os
from pathlib import Path
from collections import defaultdict


def reorganize_and_count(base_dir="/root/autodl-tmp/ResEmoteNet/rafdb_tr/train"):
    # 初始化统计字典
    original_counts = defaultdict(int)  # 原始子目录统计
    moved_files = []  # 成功移动的文件记录
    error_files = []  # 移动失败的文件记录

    # 遍历所有情感子目录
    base_path = Path(base_dir)
    for emotion_dir in base_path.iterdir():
        if emotion_dir.is_dir() and not emotion_dir.name.startswith('.'):
            emotion = emotion_dir.name
            original_count = 0

            # 处理目录中的文件
            for img_file in emotion_dir.glob('*.*'):
                if img_file.is_file():
                    try:
                        # 构造新文件名
                        new_name = f"{img_file.stem}_{emotion}{img_file.suffix}"
                        new_path = base_path / new_name

                        # 处理重名冲突
                        if new_path.exists():
                            version = 1
                            while new_path.exists():
                                new_name = f"{img_file.stem}_{emotion}_v{version}{img_file.suffix}"
                                new_path = base_path / new_name
                                version += 1

                        # 移动文件
                        img_file.rename(new_path)
                        original_count += 1
                        moved_files.append(new_path.name)
                    except Exception as e:
                        error_files.append((img_file.name, str(e)))

            original_counts[emotion] = original_count

    # 反馈统计（移动后验证）
    feedback_counts = defaultdict(int)
    for file in base_path.glob('*.*'):
        if file.is_file() and file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            try:
                # 解析情绪名称（最后一段下划线分割）
                emotion = file.stem.split('_')[-1]
                feedback_counts[emotion] += 1
            except IndexError:
                error_files.append((file.name, "文件名格式错误"))

    return {
        "original": dict(original_counts),
        "feedback": dict(feedback_counts),
        "errors": error_files,
        "moved_count": len(moved_files),
        "total_files": sum(feedback_counts.values())
    }


if __name__ == "__main__":
    result = reorganize_and_count()

    # 打印统计报表
    print("原始目录统计：")
    for emotion, count in result["original"].items():
        print(f"{emotion}: {count} 张")

    print("\n移动后验证统计：")
    for emotion, count in result["feedback"].items():
        print(f"{emotion}: {count} 张")

    print(f"\n总处理文件数: {result['moved_count']}")
    print(f"当前目录总文件数: {result['total_files']}")

    if result["errors"]:
        print("\n错误日志：")
        for error in result["errors"]:
            print(f"文件 {error[0]} 处理失败：{error[1]}")