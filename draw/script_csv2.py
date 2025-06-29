import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
from urllib3.exceptions import InsecureRequestWarning
from approach.baseline_Ghost import Baseline_Ghost
# 忽略不安全请求警告
warnings.simplefilter('ignore', InsecureRequestWarning)

# from approach.DA_EmoNet2 import DA_EmoNet  # 包含 Dual_ResBlock 的完整 Dual_EmoNet
from approach.baseline_res18 import ResNet18, ResNet34
# -------------------- 设备配置 --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 加载模型
model = ResNet18().to(device)
# -------------------- 选择模型文件 --------------------
#MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/results/20250312-202814/models"
MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models"
# MODEL_FILE = "baseline_res18_no_aug_best_test_model.pth"  # 🔴 你可以改成 "val.pth" 来加载另一个模型
MODEL_FILE = "baseline_Ghost_best_val_model.pth"  # 🔴 你可以改成 "val.pth" 来加载另一个模型
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件 {model_path} 不存在，请检查路径！")

print(f"✅ 加载模型: {model_path}")

# -------------------- 加载模型 --------------------
model = Baseline_Ghost().to(device)  # 创建模型实例并移至指定设备
checkpoint = torch.load(model_path, map_location=device)  # 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
model.eval()  # 设置为评估模式

# 定义图像预处理流程
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道灰度图
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用ImageNet标准化参数
                         std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    """
    对单张图像进行分类
    Args:
        image_path: 图像文件路径
    Returns:
        rounded_scores: 各个类别的概率得分（保留2位小数）
    """
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 使用模型进行推理
    with torch.no_grad():  # 不计算梯度，节省内存
        output = model(image)
        prob = F.softmax(output, dim=1)  # 转换为概率分布

    # 处理输出结果
    scores = prob.cpu().numpy().flatten()  # 转换为numpy数组并展平
    rounded_scores = [round(score, 2) for score in scores]  # 保留2位小数
    return rounded_scores


def process_folder(folder_path):
    """
    处理文件夹中的所有图像
    Args:
        folder_path: 图像文件夹路径
    Returns:
        results: 包含所有图像分类结果的列表
    """
    results = []
    # 遍历文件夹中的所有图像
    for img_filename in os.listdir(folder_path):
        # 只处理图像文件
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_filename)
            scores = classify_image(img_path)
            # 将文件路径和得分组合为一行
            results.append([img_path] + scores)
    return results
def main(folder_path, output_folder):
    """
    主函数：处理文件夹并保存结果

    Args:
        folder_path: 待处理的图像文件夹路径
        output_folder: CSV 文件保存的目标文件夹路径
    """
    # 获取所有图像的分类结果
    results = process_folder(folder_path)

    # 定义CSV文件头
    header = ['filepath', 'happy', 'surprise', 'sad', 'anger',
              'disgust', 'fear', 'neutral']

    # 如果输出目录不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构造CSV文件完整路径
    csv_path = os.path.join(output_folder, '93.5_classification_scores.csv')

    # 将结果写入CSV文件
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入表头
        writer.writerows(results)  # 写入数据

    print(f"结果已保存到: {csv_path}")

# 执行预测，需要替换为实际的测试文件夹路径和输出文件夹路径
# 例如：
# folder_path = '/root/autodl-tmp/ResEmoteNet/rafdb_tr/test'
# output_folder = MODEL_DIR  (或其他指定的路径)
main('/root/autodl-tmp/ResEmoteNet/rafdb_tr/test', MODEL_DIR)