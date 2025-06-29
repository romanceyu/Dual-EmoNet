import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 注意力机制升级：用ECA替换SE模块：计算更高效，性能更好
# 增加空间注意力：同时关注通道和空间维度的重要特征
# 好处：更精确地定位和增强表情相关的关键区域

# 残差块改进：添加空洞卷积分支：扩大感受野
# 集成双重注意力机制：增强特征表示
# 好处：能更好地捕捉面部表情的局部细节和全局上下文

# 多尺度特征融合：添加金字塔池化模块：融合不同尺度的特征
# 好处：更好地处理不同大小和尺度的表情特征

# 分类头优化：添加BatchNorm1d：归一化最终特征
# 优化dropout策略：更好的正则化
# 好处：提高分类准确率和模型泛化能力

class GhostModule(nn.Module):
    """Ghost模块 - 用更少的参数实现类似的特征"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class ECABlock(nn.Module):
    """Efficient Channel Attention block - 比SE更高效的通道注意力"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 局部跨通道交互
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块 - 关注重要的空间区域"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class DualAttention(nn.Module):
    """组合ECA和空间注意力"""
    def __init__(self, channels):
        super().__init__()
        self.eca = ECABlock(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.eca(x)
        x = self.spatial(x)
        return x

class StochasticDepth(nn.Module):
    """随机深度模块 - 训练时随机跳过一些层"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        if torch.rand(1).item() < self.p:
            return torch.zeros_like(x)
        return x / (1 - self.p)


class ImprovedResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p=0.2):
        super().__init__()
        # 使用Ghost模块替换普通卷积
        self.ghost1 = GhostModule(in_ch, out_ch, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.ghost2 = GhostModule(out_ch, out_ch, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.attention = DualAttention(out_ch)
        self.stochastic_depth = StochasticDepth(p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                GhostModule(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.ghost1(x)))
        out = self.bn2(self.ghost2(out))
        out = self.attention(out)
        out = self.stochastic_depth(out)
        out += self.shortcut(x)
        return F.relu(out)


class ArcMarginProduct(nn.Module):
    """ArcFace分类头"""

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class PyramidPooling(nn.Module):
    """金字塔池化模块 - 多尺度特征融合"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pools = []
        for s in [1, 2, 3, 6]:
            self.pools.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.pools = nn.ModuleList(self.pools)

    def forward(self, x):
        size = x.size()[2:]

        outputs = [x]
        for pool in self.pools:
            out = pool(x)
            out = F.interpolate(out, size, mode='bilinear',
                                align_corners=True)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class DA_EmoNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # 特征提取主干 - 使用Ghost模块
        self.conv1 = GhostModule(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = GhostModule(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = GhostModule(128, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)

        # 改进的残差块 - 添加随机深度
        self.res_block1 = ImprovedResidualBlock(256, 512, stride=2, p=0.1)
        self.res_block2 = ImprovedResidualBlock(512, 1024, stride=2, p=0.15)
        self.res_block3 = ImprovedResidualBlock(1024, 2048, stride=2, p=0.2)

        # 改进的金字塔池化 - 减少通道数
        self.pyramid = PyramidPooling(2048, 256)  # 从512减少到256

        # 特征融合
        total_channels = 2048 + 256 * 4  # 原特征 + 金字塔池化特征

        # 分类头 - 使用ArcFace
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.arc_margin = ArcMarginProduct(512, num_classes)

    def forward(self, x, labels=None):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))

        # 残差特征提取
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # 多尺度特征融合
        x = self.pyramid(x)

        # 全局池化
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # 特征提取和ArcFace分类
        features = self.feature_extractor(x)
        if self.training:
            return self.arc_margin(features, labels)
        return self.arc_margin(features)