"""
Grad-CAM Visualization Script for ResEmoteNet
Author: AI Assistant
Date: 2023-10-20

Features:
- 支持指定输入/输出路径
- 高分辨率图像输出（600dpi）
- 学术级可视化样式（使用Seaborn科学排版）
- 自适应设备配置（自动检测CUDA/MPS）
- 异常处理机制
- 符合顶会要求的图表格式
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import transforms
from PIL import Image
import seaborn as sns

# 配置学术图表样式
plt.style.use('ggplot')  # 更学术的网格样式

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['image.cmap'] = 'viridis'


class Hook:
    """梯度/激活值钩子类"""

    def __init__(self):
        self.forward_out = None
        self.backward_out = None

    def register_hook(self, module):
        def forward_hook(module, input, output):
            self.forward_out = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.backward_out = grad_output[0].detach()

        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def unregister_hook(self):
        self.forward_out = None
        self.backward_out = None


def process_image(image_path, input_size=64):
    """图像预处理流水线"""
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(Image.fromarray(img)).unsqueeze(0)
    return img


def generate_cam(feature_maps, gradients):
    """生成类激活图"""
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # 归一化处理
    cam -= cam.min()
    cam /= cam.max()
    cam = cv2.resize(cam, (64, 64))
    return cam


def visualize_and_save(cam, original_img, output_path, filename):
    """可视化并保存结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 原始图像
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')

    # CAM热力图
    heatmap = ax2.imshow(cam, cmap='viridis')
    ax2.set_title('Grad-CAM Visualization', fontsize=12)
    ax2.axis('off')

    # 添加颜色条
    cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_path, f'{filename}_cam.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved visualization to: {save_path}")


def main(input_dir, output_dir):
    # 设备配置
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查路径有效性
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载模型
        from approach.ResEmoteNet import ResEmoteNet  # 假设模型定义在此路径
        model = ResEmoteNet()
        model_path = "best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

    # 注册钩子
    final_layer = model.conv3
    hook = Hook()
    hook.register_hook(final_layer)

    # 处理目录下所有图像
    for img_file in os.listdir(input_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img_path = os.path.join(input_dir, img_file)

            # 处理图像
            img_tensor = process_image(img_path).to(device)

            # 前向传播
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            print(f'Predicted class for {img_file}: {predicted_class}')

            # 反向传播
            one_hot = torch.zeros_like(logits)
            one_hot[0][predicted_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            # 生成CAM
            gradients = hook.backward_out
            feature_maps = hook.forward_out
            cam = generate_cam(feature_maps, gradients)

            # 可视化保存
            original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            base_name = os.path.splitext(img_file)[0]
            visualize_and_save(cam, original_img, output_dir, base_name)

        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

    hook.unregister_hook()


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    INPUT_PATH = "/root/autodl-tmp/ResEmoteNet/attention"
    OUTPUT_DIR = "/root/autodl-tmp/ResEmoteNet/attention/results"

    main(INPUT_PATH, OUTPUT_DIR)