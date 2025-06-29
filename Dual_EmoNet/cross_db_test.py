import os
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
from urllib3.exceptions import InsecureRequestWarning

# 忽略不安全请求警告
warnings.simplefilter('ignore', InsecureRequestWarning)

# 导入自定义的情绪识别模型
from approach.DA_EmoNet2 import ImprovedResEmoteNet

# -------------------- 设备设置 --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# -------------------- 选择模型文件 --------------------
MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250310-143306/models"
MODEL_FILE = "best_test_model.pth"  # 🔴 你可以改成 "val.pth" 来加载另一个模型

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件 {model_path} 不存在，请检查路径！")

print(f"✅ 加载模型: {model_path}")

# -------------------- 加载模型 --------------------
model = ImprovedResEmoteNet().to(device)  # 创建模型实例并移至指定设备
checkpoint = torch.load(model_path, map_location=device)  # 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
model.eval()  # 设置为评估模式
# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小为64x64
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道灰度图
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(  # 标准化，使用ImageNet预训练模型的均值和标准差
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def classify_image(image_path):
    """
    对单张图像进行情绪分类
    Args:
        image_path: 图像文件路径
    Returns:
        list: 七种情绪的概率分数，保留两位小数
    """
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并移至指定设备

    # 进行预测
    with torch.no_grad():  # 不计算梯度
        output = model(image)
        prob = F.softmax(output, dim=1)  # 使用softmax获取概率分布

    # 转换预测结果
    scores = prob.cpu().numpy().flatten()  # 转移到CPU并展平
    rounded_scores = [round(score, 2) for score in scores]  # 保留两位小数
    return rounded_scores


def process_folder(folder_path):
    """
    处理文件夹中的所有图像

    Args:
        folder_path: 图像文件夹路径

    Returns:
        list: 包含所有图像路径和对应预测分数的列表
    """
    results = []
    for img_filename in os.listdir(folder_path):
        # 只处理图像文件
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_filename)
            scores = classify_image(img_path)
            results.append([img_path] + scores)
    return results



def main(folder_path, output_folder):
    """
    主函数：处理文件夹并保存结果

    Args:
        folder_path: 待处理的图像文件夹路径
        output_folder: 输出文件保存的文件夹路径
    """
    # 处理文件夹中的所有图像
    results = process_folder(folder_path)

    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 创建文件夹

    # 设置输出文件的完整路径
    output_file = os.path.join(output_folder, 'testfile.csv')

    # 定义CSV文件表头
    header = ['filepath', 'happy', 'surprise', 'sad',
              'anger', 'disgust', 'fear', 'neutral']

    # 将结果写入CSV文件
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

    print(f"结果已保存到: {output_file}")


# 执行预测，需要替换为实际的测试文件夹路径
main('/root/autodl-tmp/ResEmoteNet/rafdb_tr/test',MODEL_DIR)