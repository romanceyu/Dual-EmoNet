import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import matplotlib as mpl

# ==================== 学术图表全局配置 ====================
plt.style.use('ggplot')  # 更学术的网格样式
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',  # 开源无衬线字体
    'axes.titlesize': 16,  # IEEE Trans 推荐标题字号
    'axes.labelsize': 14,  # 坐标轴标签字号
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,  # 交互显示清晰度
    'savefig.dpi': 600,  # 印刷级分辨率
    'image.cmap': 'viridis',  # 色盲友好色系
    'legend.frameon': True,  # 图例边框
    'figure.constrained_layout.use': True,
    'figure.figsize': (18, 6)  # 标准三栏布局尺寸
})


class GradCAMGenerator:
    """面向学术出版的 Grad-CAM 可视化工具

    主要优化特性:
    - 支持批量处理 (单图/目录)
    - 自适应设备选择 (CUDA/MPS/CPU)
    - 可配置的学术图表样式
    - 异常处理机制
    """

    def __init__(self, model_dir, model_file):
        # 设备配置
        self.device = self._get_device()
        print(f"⚙️ 运行设备: {self.device}")

        # 模型初始化
        self.model = self._load_model(model_dir, model_file)
        self.hook = None

        # 图像预处理流水线
        self.preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    class Hook:
        """梯度捕获工具 (支持多模态特征提取)"""

        def __init__(self):
            self.forward_out = None
            self.backward_out = None

        def register(self, module):
            def forward_hook(_, __, output):
                self.forward_out = output.detach()

            def backward_hook(_, grad_input, grad_output):
                self.backward_out = grad_output[0].detach()

            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)

    def _get_device(self):
        """自动选择最优计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_model(self, model_dir, model_file):
        """加载预训练模型 (含完整性校验)"""
        from approach.DA_EmoNet2 import DA_EmoNet  # 包含 Dual_ResBlock 的完整 Dual_EmoNet
        # from approach.baseline_Dual_ResBlock import Baseline_Dual_ResBlock
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件缺失: {model_path}")

        print(f"✅ 正在加载模型: {model_path}")
        model = DA_EmoNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _process_image(self, image_path):
        """图像预处理流水线 (含异常处理)"""
        try:
            img = Image.open(image_path).convert('RGB')
            original_img = np.array(img)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            return original_img, img_tensor
        except Exception as e:
            raise RuntimeError(f"图像处理失败: {image_path} | 错误: {str(e)}")

    def generate_cam(self, image_path):
        """生成 Grad-CAM 热力图"""
        try:
            # 注册钩子到最后一个卷积层
            self.hook = self.Hook()
            self.hook.register(self.model.res_block3.attention)  # 修改为 conv2

            # 图像预处理
            original_img, img_tensor = self._process_image(image_path)

            # 前向传播
            logits = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            pred_class = torch.argmax(probabilities).item()

            # 反向传播
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0][pred_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            # 计算 CAM
            weights = torch.mean(self.hook.backward_out, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * self.hook.forward_out, dim=1, keepdim=True)
            cam = torch.relu(cam).squeeze().cpu().numpy()

            # 后处理
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
            return original_img, cam, pred_class

        except Exception as e:
            print(f"🔥 Grad-CAM 生成失败: {str(e)}")
            raise

    def visualize(self, original_img, cam, output_path, filename):
        """学术级可视化输出"""
        try:
            #
            # # 自适应模糊参数调整（根据热力图尺寸动态计算）
            # h, w = cam.shape
            # base_size = min(h, w)
            #
            # # 智能参数计算（经验公式）
            # sigma_ratio = 0.015  # 基础比例系数，可调范围0.01-0.03
            # kernel_ratio = 0.04  # 核尺寸比例，可调范围0.03-0.06
            #
            # # 动态参数生成
            # sigma = max(5.0, base_size * sigma_ratio)  # 保证最小sigma=3.0
            # kernel_size = int(base_size * kernel_ratio)
            # kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 保证奇数
            #
            # # 执行自适应高斯模糊
            # blurred_cam = cv2.GaussianBlur(
            #     cam,
            #     (kernel_size, kernel_size),
            #     sigmaX=sigma,
            #     sigmaY=sigma
            # )
            # # 创建叠加图像
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
            superimposed = cv2.addWeighted(
                cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.5,
                heatmap, 0.5, 0
            )

            # 创建画布
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('', y=0.95, fontsize=18, weight='bold')

            # 子图1: 原图
            axes[0].imshow(original_img)
            axes[0].set_title('(a) Input Image', fontsize=14, pad=10)
            axes[0].axis('off')

            # 子图2: 热力图
            im = axes[1].imshow(cam, cmap='viridis')
            axes[1].set_title('(b) Activation Map', fontsize=14, pad=10)
            axes[1].axis('off')
            cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Activation Intensity', rotation=270, labelpad=15)

            # 子图3: 叠加效果
            axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            axes[2].set_title('(c) Overlay Result', fontsize=14, pad=10)
            axes[2].axis('off')

            # 保存结果
            output_file = os.path.join(output_path, f"{filename}_cam.png")
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=600)
            plt.close()
            print(f"💾 结果保存至: {output_file}")

        except Exception as e:
            print(f"🔥 可视化保存失败: {str(e)}")
            raise

def main():
    # ==================== 配置参数 ====================
    # MODEL_PATH = "/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models/baseline_Dual_ResBlock_best_test_model.pth"
    # / best_test_model
    MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models"
    # MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models"
    MODEL_FILE = "best_test_model.pth"
    INPUT_PATH = "/root/autodl-tmp/ResEmoteNet/20-attention"
    OUTPUT_DIR = "/root/autodl-tmp/ResEmoteNet/attention/results2"
    try:
        # 初始化生成器
        generator = GradCAMGenerator(MODEL_DIR, MODEL_FILE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 处理输入路径
        if os.path.isfile(INPUT_PATH):
            print(f"🔍 处理单张图像: {INPUT_PATH}")
            original_img, cam, _ = generator.generate_cam(INPUT_PATH)
            filename = os.path.splitext(os.path.basename(INPUT_PATH))[0]
            generator.visualize(original_img, cam, OUTPUT_DIR, filename)

        elif os.path.isdir(INPUT_PATH):
            print(f"📂 批量处理目录: {INPUT_PATH}")
            for img_file in sorted(os.listdir(INPUT_PATH)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(INPUT_PATH, img_file)
                    print(f"  ➤ 处理: {img_file}")
                    original_img, cam, _ = generator.generate_cam(img_path)
                    filename = os.path.splitext(img_file)[0]
                    generator.visualize(original_img, cam, OUTPUT_DIR, filename)

        else:
            raise ValueError(f"无效输入路径: {INPUT_PATH}")

    except Exception as e:
        print(f"❌ 运行终止: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()