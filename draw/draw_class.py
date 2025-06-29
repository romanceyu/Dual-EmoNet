import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 1) 全局风格
plt.rc('font', family='DejaVu Sans', size=12)
plt.rc('axes', grid=False)  # We'll add custom grids
plt.rc('grid', linestyle='--', alpha=0.4)

# 2) 输出目录
results_dir = r'F:\Desktop\results'
os.makedirs(results_dir, exist_ok=True)

# 3) 标签顺序
labels = ['0 Happy', '1 Surprise', '2 Sad', '3 Angry', '4 Disgust', '5 Fear', '6 Neutral']

# 4) 原始计数
raf_orig_train    = [1290, 281, 717, 4772, 1982, 705, 2524]
raf_orig_test     = [329,   74, 160, 1185,  478, 162,  680]
affect_orig_train = [74874, 134415, 25459, 14090, 6378, 3803, 24882]
affect_orig_test  = [500] * 7
ferp_orig_train   = [8740, 7287, 3149, 3014, 2100, 119, 532, 119]  # 包含 'Contempt'
ferp_orig_test    = [1090, 893, 396, 384, 273, 18, 83, 16]

# 5) 重排序到 [Happy, Surprise, Sad, Angry, Disgust, Fear, Neutral]
raf_train    = [raf_orig_train[i]    for i in [3,0,4,5,2,1,6]]
raf_test     = [raf_orig_test[i]     for i in [3,0,4,5,2,1,6]]
affect_train = [affect_orig_train[i] for i in [1,3,2,6,5,4,0]]
affect_test  = [affect_orig_test[i]  for i in [1,3,2,6,5,4,0]]
# FERPlus 丢弃 'Contempt'（index 7）
ferp_train   = [ferp_orig_train[i]   for i in [1,2,3,4,5,6,0]]
ferp_test    = [ferp_orig_test[i]    for i in [1,2,3,4,5,6,0]]

# 6) 颜色
colors = {
    'Training': '#a6cee3',
    'Test':     '#b2df8a'
}

def plot_dist(counts_train, counts_test, title, filename):
    x         = np.arange(len(labels))
    bar_width = 0.3
    gap       = 0.1  # gap between train & test bars

    fig, ax = plt.subplots(figsize=(9,5))
    # 设置半透明白色背景
    ax.set_facecolor((1, 1, 1, 0.7))
    # 隐藏顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 绘制柱状
    ax.bar(x - (bar_width+gap)/2, counts_train, bar_width,
           label='Training', color=colors['Training'], zorder=2)
    ax.bar(x + (bar_width+gap)/2, counts_test,  bar_width,
           label='Test',     color=colors['Test'],     zorder=2)

    # 主/次刻度
    ax.minorticks_on()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    # 网格
    ax.grid(which='major', linestyle='--', linewidth=0.7,
            color='gray', alpha=0.5, zorder=1)
    ax.grid(which='minor', linestyle=':', linewidth=0.5,
            color='gray', alpha=0.3, zorder=1)

    # 留白
    ymax = max(counts_train + counts_test)
    ax.set_ylim(0, ymax * 1.1)

    # 标注数值
    y_off = ymax * 0.005
    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + y_off,
                f'{int(h)}', ha='center', va='bottom', fontsize=9, zorder=3)

    # 坐标轴标签 & 标题
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)  # 小号字体，不旋转
    ax.set_xlabel('Expression Category')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path = os.path.join(results_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")

# 7) 绘制并保存
plot_dist(raf_train,  raf_test,  'RAF‑DB Class Distribution',     'RAFDB_distribution.png')
plot_dist(affect_train, affect_test, 'AffectNet‑7 Class Distribution', 'AffectNet7_distribution.png')
plot_dist(ferp_train, ferp_test, 'FERPlus Class Distribution',     'FERPlus_distribution.png')
