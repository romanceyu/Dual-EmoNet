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
    'image.cmap': 'jet',  # 色盲友好色系
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
        from approach.DA_EmoNet2 import ImprovedResEmoteNet

        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件缺失: {model_path}")

        print(f"✅ 正在加载模型: {model_path}")
        model = ImprovedResEmoteNet().to(self.device)
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
            self.hook.register(self.model.conv3)

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
        """优化后的学术级叠加效果"""
        try:
            # 增强CAM对比度 (非线性伽马校正)
            cam_enhanced = np.power(cam, 0.6)  # 指数小于1以增强高值区域

            # 生成高对比度热力图 (改用JET色图)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_enhanced), cv2.COLORMAP_VIRIDIS)

            # 调整叠加比例 (原图透明度降低)
            superimposed = cv2.addWeighted(
                cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.5,  # 原图占比30%
                heatmap, 0.5,  # 热力图占比70%
                0  # 亮度调节
            )

            # 后处理增强
            superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(superimposed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 自适应对比度增强
            limg = cv2.merge([clahe.apply(l), a, b])
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            # 创建专业可视化布局
            fig = plt.figure(figsize=(15, 8))
            gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05])

            # 子图1: 原始图像
            ax1 = fig.add_subplot(gs[:, 0])
            ax1.imshow(original_img)
            ax1.set_title('(a) Input Image', fontsize=14, y=0.95,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax1.axis('off')

            # 子图2: 纯热力图
            ax2 = fig.add_subplot(gs[:, 1])
            im = ax2.imshow(cam_enhanced, cmap='viridis')
            ax2.set_title('(b) Activation Map', fontsize=14, y=0.95,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax2.axis('off')

            # 子图3: 优化叠加效果
            ax3 = fig.add_subplot(gs[:, 2])
            ax3.imshow(enhanced_img)
            ax3.set_title('(c) Enhanced Overlay', fontsize=14, y=0.95,
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax3.axis('off')

            # 专业颜色条
            cax = fig.add_subplot(gs[:, 3])
            cbar = plt.colorbar(im, cax=cax, aspect=10)
            cbar.set_label('Activation Intensity', rotation=270,
                           labelpad=20, fontsize=12)
            # 保存优化结果
            output_file = os.path.join(output_path, f"{filename}_cam_optimized.png")
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=600)
            plt.close()

        except Exception as e:
            print(f"可视化优化失败: {str(e)}")
            raise

def main():
    # ==================== 配置参数 ====================
    MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models"
    MODEL_FILE = "best_test_model.pth"
    INPUT_PATH = "/root/autodl-tmp/ResEmoteNet/attention/test_1404_aligned_anger.jpg"
    OUTPUT_DIR = "/root/autodl-tmp/ResEmoteNet/attention/results"

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