import os

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from approach.DA_EmoNet2 import ImprovedResEmoteNet

# -------------------- 选择模型文件 --------------------
MODEL_DIR = "/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models"
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
img_path = '/root/autodl-tmp/ResEmoteNet/attention/test_1404_aligned_anger.jpg'

# 如果需要，可以直接用 PIL 打开图像（但下面的函数中会用 cv2 读取）
img = Image.open(img_path)
img = np.array(img)  # 变为 (H, W, C) 形状的 NumPy 数组
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 将 OpenCV 的 numpy 数组转换为 PIL 图像
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 使用传入的 image_path，而不是硬编码的 img_path
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img).unsqueeze(0)
    return img

from hook import Hook

final_layer = model.conv3
hook = Hook()
hook.register_hook(final_layer)

img_tensor = process_image(img_path).to(device)

logits = model(img_tensor)
probabilities = F.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1)

predicted_class_idx = predicted_class.item()
print(f'Predicted class: {predicted_class_idx}')

one_hot_output = torch.FloatTensor(1, probabilities.shape[1]).zero_().to(device)
one_hot_output[0][predicted_class_idx] = 1
logits.backward(one_hot_output, retain_graph=True)

gradients = hook.backward_out
feature_maps = hook.forward_out

hook.unregister_hook()

weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
cam = cam.clamp(min=0).squeeze()

cam -= cam.min()
cam /= cam.max()
cam = cam.cpu().detach().numpy()
cam = cv2.resize(cam, (64, 64))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
heatmap = np.float32(heatmap) / 255

# 调整原始图像尺寸，使其与热力图一致
img_resized = cv2.resize(img, (64, 64))
superimposed_img = heatmap * 0.5 + np.float32(img_resized) / 255
superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype(np.uint8)  # 限制到 0-255 并转换为 uint8



plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis('off')
plt.legend('原图')

plt.subplot(1, 3, 2)

cax = plt.matshow(cam, cmap='viridis', fignum=0)
plt.axis('off')
plt.colorbar(cax, ax=plt.gca(), fraction=0.045, pad=0.05)
plt.legend('heatmap')
# superimposed = cv2.addWeighted(
#                 cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR), 0.5,  # 原图占比30%
#                 heatmap, 0.5,  # 热力图占比70%
#                 0  # 亮度调节
#             )

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.axis('off')
plt.legend('叠加后')

plt.savefig('gcam_anger.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
plt.show()

# 定义保存路径
output_dir = "/root/autodl-tmp/ResEmoteNet/attention"
filename = "anger.png"
save_path = os.path.join(output_dir, filename)

# 确保目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 保存图像
# 保存高清图像
plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)

# 显示图像
plt.show()