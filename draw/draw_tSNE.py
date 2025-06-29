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
        model = ResNet18().to(self.device)
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
                # 关键修改：通过完整模型前向传播
                outputs = self.model(inputs)  # 假设模型可以直接返回特征

                # 如果模型返回的是元组，需要索引特征部分
                if isinstance(outputs, tuple):
                    feats = outputs[0]  # 假设特征在第一个返回值
                else:
                    feats = outputs

                features.append(feats.cpu().numpy())
                labels.append(targets.numpy())

        return np.concatenate(features), np.concatenate(labels)

    def visualize(self, test_dataset, save_dir, n_samples=3000, perplexity=35):

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 创建数据加载器
            dataloader = DataLoader(test_dataset,
                                    batch_size=16,
                                    shuffle=True,  # 保证随机采样
                                    num_workers=4)

            # 提取特征
            print("⏳ 正在提取特征...")
            features, labels = self._get_features(dataloader)
            print(f"提取特征数: {features.shape[0]}, 特征维度: {features.shape[1]}")
            # 随机采样
            if len(features) > n_samples:
                indices = np.random.choice(len(features), n_samples, replace=False)
                features = features[indices]
                labels = labels[indices]
            # 在visualize()方法中添加：
            print("📊 特征矩阵维度:", features.shape)
            # PCA预降维加速
            print("🔧 正在执行PCA降维...")
            # 修改PCA初始化代码为
            pca = PCA(n_components=min(50, features.shape[1] - 1))  # 自动适配最大可用维度
            features_pca = pca.fit_transform(features)

            # t-SNE降维
            print("🌀 正在执行t-SNE...")
            tsne = TSNE(n_components=2,
                        perplexity=perplexity,
                        learning_rate=200,
                        random_state=42)
            tsne_results = tsne.fit_transform(features_pca)

            # 可视化
            print("🎨 正在生成可视化...")
            plt.figure(figsize=(14, 10))

            # 定义颜色和标签
            emotion_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']

            # 绘制散点
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                                  c=labels, cmap=ListedColormap(colors),
                                  alpha=0.8, edgecolors='k', linewidth=0.3,
                                  s=60, zorder=2)

            # 添加图例
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          label=emotion_names[i],
                                          markerfacecolor=colors[i],
                                          markersize=12, markeredgecolor='k')
                               for i in range(7)]
            plt.legend(handles=legend_elements,
                       title='Emotion Classes',
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left',
                       borderaxespad=0.,
                       frameon=False)

            # 坐标轴优化
            plt.xlabel('t-SNE Dimension 1', labelpad=12, fontweight='bold')
            plt.ylabel('t-SNE Dimension 2', labelpad=12, fontweight='bold')
            plt.title('Feature Embedding Visualization (Test Set)',
                      pad=20, fontsize=18, fontweight='bold')

            # 网格线增强可读性
            plt.grid(True, linestyle='--', alpha=0.6, zorder=1)
            # 保存结果
            # 双格式保存
            base_path = os.path.join(save_dir, 'tsne_visualization')
            plt.savefig(f"{base_path}.png", bbox_inches='tight')
            plt.savefig(base_path, bbox_inches='tight')
            print(f"💾 结果已保存至: {base_path}")
            plt.close()

        except Exception as e:
            print(f"❌ 可视化失败: {str(e)}")
            raise


if __name__ == "__main__":
    # ==================== 配置参数 ====================
    # MODEL_PATH = "/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models/best_test_model.pth"
    MODEL_PATH = "/root/autodl-tmp/ResEmoteNet/runs/20250404-142614/models/baseline_res18_no_aug_best_test_model.pth"
    TEST_CSV = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/test_labels.csv"
    TEST_IMG_DIR = "/root/autodl-tmp/ResEmoteNet/rafdb_tr/test"
    models_dir = os.path.dirname(MODEL_PATH)
    base_dir = os.path.dirname(models_dir)
    SAVE_DIR = os.path.join(base_dir, "t-SNE")

    # 初始化可视化器
    visualizer = TSNEVisualizer(MODEL_PATH)

    # 创建测试数据集
    test_dataset = Four4All(
        csv_file=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=visualizer.transform
    )

    # 执行可视化
    visualizer.visualize(
        test_dataset=test_dataset,
        save_dir=SAVE_DIR,
        n_samples=3000,  # 建议采样量
        perplexity=35  # 根据数据分布调整
    )