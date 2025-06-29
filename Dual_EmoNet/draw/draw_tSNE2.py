import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from torchvision import transforms
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from approach.DA_EmoNet2 import DA_EmoNet  # 包含 Dual_ResBlock 的完整 Dual_EmoNet
from approach.baseline_res18 import ResNet18, ResNet34
from approach.baseline_Dual_ResBlock import Baseline_Dual_ResBlock
from get_dataset import Four4All

# ==================== 学术可视化配置 ====================
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.format': 'svg',
    'svg.fonttype': 'none'  # 确保文字可编辑
})

class TSNEVisualizer:
    def __init__(self, model_path, num_classes=7):
        # 设备配置
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # 模型加载
        self.model = self._load_model(model_path)
        self.num_classes = num_classes

        # 数据转换
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path):
        """安全加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {model_path}")

        print(f"✅ 正在加载模型: {model_path}")
        model = DA_EmoNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _get_features(self, dataloader):
        """批量提取特征"""
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)  # 假设模型返回特征
                if isinstance(outputs, tuple):
                    feats = outputs[0]
                else:
                    feats = outputs
                features.append(feats.cpu().numpy())
                labels.append(targets.numpy())

        return np.concatenate(features), np.concatenate(labels)

    def visualize(self, test_dataset, save_dir, n_samples=3000, perplexity=35):
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        try:
            dataloader = DataLoader(test_dataset,
                                    batch_size=16,
                                    shuffle=True,
                                    num_workers=4)

            print("⏳ 正在提取特征...")
            features, labels = self._get_features(dataloader)
            print(f"提取特征数: {features.shape[0]}, 特征维度: {features.shape[1]}")
            if len(features) > n_samples:
                indices = np.random.choice(len(features), n_samples, replace=False)
                features = features[indices]
                labels = labels[indices]
            print("📊 特征矩阵维度:", features.shape)

            print("🔧 正在执行PCA降维...")
            pca = PCA(n_components=min(50, features.shape[1] - 1))
            features_pca = pca.fit_transform(features)

            print("🌀 正在执行t-SNE...")
            tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, random_state=42)
            tsne_results = tsne.fit_transform(features_pca)

            print("🎨 正在生成可视化...")
            plt.figure(figsize=(14, 10))
            ax = plt.gca()
            # 设置背景色为白色半透明
            ax.set_facecolor((1, 1, 1, 0.7))

            # 开启次刻度
            ax.minorticks_on()

            # 设置主网格：使用柔和的灰色虚线，线宽适中，透明度适中
            ax.grid(which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.5, zorder=1)
            # 设置次网格：使用更细的虚线，颜色相同但透明度更低
            ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.3, zorder=1)

            # 正确的情绪顺序，根据映射:
            # "happy": 0, "surprise": 1, "sad": 2, "angry": 3, "disgust": 4, "fear": 5, "neutral": 6
            emotions = ['Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral']

            # 设置柔和 pastel 色系，对应顺序
            pastel_colors = ['#a6cee3',  # Happy
                             '#b2df8a',  # Surprise
                             '#fb9a99',  # Sad
                             '#fdbf6f',  # Angry
                             '#cab2d6',  # Disgust
                             '#ffffcc',  # Fear
                             '#fddaec']  # Neutral

            # 在图例中也使用这个正确顺序
            emotion_names = emotions  # 或直接用 emotions 变量

            cmap = ListedColormap(pastel_colors)

            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                                  c=labels, cmap=cmap, alpha=0.8, edgecolors='k', linewidth=0.3,
                                  s=60, zorder=2)

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          label=emotion_names[i],
                                          markerfacecolor=pastel_colors[i],
                                          markersize=12, markeredgecolor='k')
                               for i in range(len(emotion_names))]
            # 图例放置在右下角，防止遮挡散点
            plt.legend(handles=legend_elements, title='Emotion Classes',
                       loc="lower right", bbox_to_anchor=(0.98, 0.02),
                       borderaxespad=0.5, frameon=False)

            plt.xlabel('t-SNE Dimension 1', labelpad=12, fontweight='bold')
            plt.ylabel('t-SNE Dimension 2', labelpad=12, fontweight='bold')
            plt.title('Feature Embedding Visualization (Test Set)',
                      pad=20, fontsize=18, fontweight='bold')

            plt.tight_layout()
            base_path = os.path.join(save_dir, 'tsne_visualization')
            plt.savefig(f"{base_path}.png", bbox_inches='tight', dpi=300)
            print(f"💾 结果已保存至: {base_path}.png")
            plt.close()

        except Exception as e:
            print(f"❌ 可视化失败: {str(e)}")
            raise


if __name__ == "__main__":
    MODEL_PATH = "/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models/best_test_model.pth"
    # MODEL_PATH = "/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models/baseline_Dual_ResBlock_best_test_model.pth"
    TEST_CSV = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/test_labels.csv"
    TEST_IMG_DIR = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/test"
    models_dir = os.path.dirname(MODEL_PATH)
    base_dir = os.path.dirname(models_dir)
    SAVE_DIR = os.path.join(base_dir, "t-SNE")

    visualizer = TSNEVisualizer(MODEL_PATH)

    test_dataset = Four4All(
        csv_file=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=visualizer.transform
    )

    visualizer.visualize(
        test_dataset=test_dataset,
        save_dir=SAVE_DIR,
        n_samples=3000,
        perplexity=35
    )
