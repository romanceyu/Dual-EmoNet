import os
from tqdm import tqdm
from torchvision.utils import make_grid
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import seaborn as sns
from thop import profile  # 用于计算FLOPs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

# 自定义模块
from approach.DA_EmoNet import DA_EmoNet
from get_dataset import Four4All

def set_seed(seed=0):
    random.seed(seed)  # Python的random模块
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU单卡
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU多卡
    print(seed)
    # 额外设置以提高确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(0)
# -------------------- TensorBoard初始化 --------------------
# 创建TensorBoard日志目录，使用当前时间戳作为目录名
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'runs/{current_time}'
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs,Modules and result_csv will be saved to: {log_dir}")

# 创建保存目录
results_dir = f"./runs/{current_time}"
os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

# -------------------- 设备设置 --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# -------------------- 新增全局变量 --------------------
# 初始化评测指标列表
test_f1_scores = []
test_precisions = []
test_recalls = []
val_f1_scores = []
val_precisions = []
val_recalls = []
start_time = time.time()  # 记录训练开始时间

# -------------------- 计算FLOPs的函数 --------------------
def calculate_model_resources(model, input_size):
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return params, flops

class DynamicLabelSmoothing(nn.Module):
    """动态标签平滑"""
    def __init__(self, classes, min_smoothing=0.1, max_smoothing=0.2):
        super().__init__()
        self.classes = classes
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.epoch = 0

    def forward(self, pred, target):
        # 根据训练进度动态调整平滑程度
        smoothing = self.max_smoothing - (self.max_smoothing - self.min_smoothing) * min(self.epoch / 30, 1.0)

        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
    # 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强设置
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    # -------------------- 加载数据集 --------------------
train_dataset = Four4All(
    csv_file='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/train_labels.csv',
    img_dir='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/train',
    transform=transform_train
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=16,
                        persistent_workers=True, pin_memory=True, prefetch_factor=4)
val_dataset = Four4All(
    csv_file='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/val_labels.csv',
    img_dir='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/val',
    transform=transform_test
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16,
                        persistent_workers=True, pin_memory=True, prefetch_factor=4)
test_dataset = Four4All(
    csv_file='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/test_labels.csv',
    img_dir='F:/Desktop/sci/project/ResEmoteNet-test/ResEmoteNet/rafdb/test',
    transform=transform_test
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16,
                        persistent_workers=True, pin_memory=True, prefetch_factor=4)

    # 初始化模型
model = DA_EmoNet().to(device)
    # 初始化损失函数和数据增强
criterion = DynamicLabelSmoothing(classes=7)

# 获取一个批次的数据并添加到TensorBoard
images, labels = next(iter(train_loader))
# 添加示例图片到TensorBoard
grid = make_grid(images)
writer.add_image('示例训练图片', grid, 0)
print("已将示例图片添加到TensorBoard")


# -------------------- 模型初始化 --------------------
# 首先集成ECA注意力
# model = ImprovedResEmoteNet().to(device)
# 验证效果后再添加空间注意力
# 最后加入金字塔池化模块
# model = ResEmoteNetV2()

# 将模型结构添加到TensorBoard
dummy_input = torch.randn(1, 3, 64, 64).to(device)
writer.add_graph(model, dummy_input)
print("已将模型结构添加到TensorBoard")

# 打印模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# -------------------- 超参数设置 --------------------
# AdamW+ReduceLROnPlateau+ECA = 96%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4*2, weight_decay=1e-5)

# # 使用学习率调度器
# AdamW+ReduceLROnPlateau(05+3)+ECA = 96%:
# 新的最佳测试集准确率: 0.96217802
# 新的最佳验证集准确率: 0.92376431 (提升: 0.00027925)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
    verbose=True,
    min_lr=1e-6
)


# 早停机制参数
patience = 15
patience_counter = 0
epoch_counter = 0
num_epochs = 90
# 初始化最佳准确率记录
best_test_acc = 0
best_val_acc = 0

# 记录训练指标
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
test_losses = []
test_accuracies = []
test_improvements = []  # 记录测试集精度提升
val_improvements = []   # 记录验证集精度提升
learning_rates = []

# 记录更多评测指标：使用列表保存每个epoch的详细预测结果（验证集和测试集）
val_reports = []  # 存储验证集的分类报告
test_reports = []  # 存储测试集的分类报告

def dynamic_grad_clip(parameters, max_norm, epoch):
    """动态梯度裁剪"""
    # 根据训练进度动态调整裁剪阈值
    clip_factor = 1.0 - 0.5 * min(epoch / 30, 1.0)  # 随训练进行逐渐放松限制
    current_norm = max_norm * clip_factor
    torch.nn.utils.clip_grad_norm_(parameters, current_norm)

# 在训练循环开始前添加模型保存设置
model_save_dir = os.path.join(log_dir, 'models')
os.makedirs(model_save_dir, exist_ok=True)

# 分别设置测试集和验证集最佳模型的保存路径
best_test_model_path = os.path.join(model_save_dir, 'best_test_model.pth')
best_val_model_path = os.path.join(model_save_dir, 'best_val_model.pth')

# -------------------- 训练循环 --------------------
for epoch in range(num_epochs):
    # ---------- 训练阶段 ----------
    model.train()
    criterion.epoch = epoch  # 更新动态标签平滑的epoch
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用tqdm显示进度条
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        dynamic_grad_clip(model.parameters(), max_norm=1.0, epoch=epoch)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 每100个批次记录一次训练过程中的损失
        if batch_idx % 100 == 99:
            writer.add_scalar('训练/批次损失',
                              running_loss / 100,
                              epoch * len(train_loader) + batch_idx)
            running_loss = 0.0

    # 计算训练指标
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 添加训练指标到TensorBoard
    writer.add_scalar('损失/训练', train_loss, epoch)
    writer.add_scalar('准确率/训练', train_acc, epoch)

    # ---------- 测试阶段 ----------
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', test_acc, epoch)

    # 计算测试集的额外评测指标
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_precision = precision_score(test_labels, test_preds, average='macro')
    test_recall = recall_score(test_labels, test_preds, average='macro')
    test_f1_scores.append(test_f1)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)

    writer.add_scalar('Test/F1', test_f1, epoch)
    writer.add_scalar('Test/Precision', test_precision, epoch)
    writer.add_scalar('Test/Recall', test_recall, epoch)
    test_report = classification_report(test_labels, test_preds, output_dict=True)
    test_reports.append(test_report)
    test_improvement = 0.0  # 初始化提升值为0
    # 保存测试集最佳模型
    if test_acc > best_test_acc:
        test_improvement = test_acc - best_test_acc
        best_test_acc = test_acc
        if epoch >= 30:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_improvement': test_improvement,
                'test_loss': test_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, best_test_model_path)
    test_improvements.append(test_improvement)

    # ---------- 验证集评估 ----------
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # 添加验证指标到TensorBoard
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Val/Accuracy', val_acc, epoch)
    # 计算额外评测指标（验证集）
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro')
    val_recall = recall_score(val_labels, val_preds, average='macro')
    val_f1_scores.append(val_f1)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)

    writer.add_scalar('Val/F1', val_f1, epoch)
    writer.add_scalar('Val/Precision', val_precision, epoch)
    writer.add_scalar('Val/Recall', val_recall, epoch)
    # 可打印详细分类报告（可选）
    val_report = classification_report(val_labels, val_preds, output_dict=True)
    val_reports.append(val_report)
    # 记录当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    writer.add_scalar('学习率', current_lr, epoch)

    # 更新学习率
    scheduler.step(val_acc)
    val_improvement = 0.0  # 默认提升为0
    # 打印当前epoch的训练结果
    print(f"当前学习率: {current_lr}")
    print(f"训练损失: {train_loss:.8f}, 训练准确率: {train_acc:.8f}")
    print(f"测试损失: {test_loss:.8f}, 测试准确率: {test_acc:.8f}  (提升: {test_improvement:.8f})")
    print(f"验证损失: {val_loss:.8f}, 验证准确率: {val_acc:.8f}")
    print(f"新的最佳测试集准确率: {best_test_acc:.8f} ")
    # 生成 `classification_report` 并保存到 txt
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    report_path = os.path.join(results_dir, 'logs', 'logs_report.txt')

    epoch_counter += 1


    # ---------- 早停机制 ----------
    if val_acc > best_val_acc:
        val_improvement = val_acc - best_val_acc
        best_val_acc = val_acc

        patience_counter = 0
        # 保存最佳模型
        if epoch >=30:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_improvement': val_improvement,
                'best_val_acc': best_val_acc,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, best_val_model_path)
        print(f"新的最佳验证集准确率: {best_val_acc:.8f} (提升: {val_improvement:.8f})")
    else:
        patience_counter += 1
        print(f"验证准确率未提升: {patience_counter} 轮")

    if patience_counter > patience:
        print("触发早停机制，训练结束")
        print(f"\n最终结果:")
        print(f"最佳测试集准确率: {best_test_acc:.4f}, 模型路径: {best_test_model_path}")
        print(f"最佳验证集准确率: {best_val_acc:.4f}, 模型路径: {best_val_model_path}")
        break
    val_improvements.append(val_improvement)

    # 追加写入
    with open(report_path, 'a+', encoding='utf-8') as f:
        f.write(f"\n===== Epoch {epoch + 1} =====\n")

        f.write("\n--- Test Report ---\n")
        f.write(classification_report(test_labels, test_preds))

        f.write("\n--- Validation Report ---\n")
        f.write(classification_report(val_labels, val_preds))

        f.write(f"当前学习率: {current_lr:.8e}\n")  # 使用科学计数法格式化学习率
        f.write(f"训练损失: {train_loss:.8f}, 训练准确率: {train_acc:.8f}\n")
        f.write(f"测试损失: {test_loss:.8f}, 测试准确率: {test_acc:.8f}\n")
        f.write(f"验证损失: {val_loss:.8f}, 验证准确率: {val_acc:.8f}\n")
        f.write(f"新的最佳测试集准确率: {best_test_acc:.8f}\n")
        f.write(f"新的最佳验证集准确率: {best_val_acc:.8f}\n")
        f.write("\n" + "=" * 50 + "\n")

# 关闭TensorBoard写入器
writer.close()

def save_training_results(epoch_counter, train_losses, test_losses, val_losses,
                          train_accuracies, test_accuracies, val_accuracies,
                          test_f1, test_precision, test_recall,  # 新增指标
                          val_f1, val_precision, val_recall,  # 新增指标
                          test_improvements, val_improvements, learning_rates,
                          model, optimizer, transform, current_time,
                          train_loader, test_loader, val_loader,
                          num_epochs, patience):

    # 创建保存目录
    results_dir = f"./runs/{current_time}"
    os.makedirs(os.path.join(results_dir, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'config'), exist_ok=True)

    min_length = min(len(train_losses), len(test_losses), len(val_losses),
                     len(train_accuracies), len(test_accuracies), len(val_accuracies),
                     len(test_improvements), len(val_improvements), len(learning_rates),
                     len(test_f1), len(test_precision), len(test_recall),
                     len(val_f1), len(val_precision), len(val_recall))

    dynamic_data = {
        'Epoch': range(1, min_length + 1),
        'Train_Loss': train_losses[:min_length],
        'Test_Loss': test_losses[:min_length],
        'Val_Loss': val_losses[:min_length],
        'Train_Accuracy': train_accuracies[:min_length],
        'Test_Accuracy': test_accuracies[:min_length],
        'Test_F1_Score': test_f1[:min_length],  # 修正
        'Test_Precision': test_precision[:min_length],  # 修正
        'Test_Recall': test_recall[:min_length],  # 修正
        'Test_Accuracy_Improvement': test_improvements[:min_length],
        'Val_Accuracy': val_accuracies[:min_length],
        'Val_F1_Score': val_f1[:min_length],  # 修正
        'Val_Precision': val_precision[:min_length],  # 修正
        'Val_Recall': val_recall[:min_length],  # 修正
        'Val_Accuracy_Improvement': val_improvements[:min_length],
        'Learning_Rate': learning_rates[:min_length]
    }

    dynamic_df = pd.DataFrame(dynamic_data)
    csv_path = os.path.join(results_dir, 'csv', 'training_metrics.csv')
    dynamic_df.to_csv(csv_path, index=False)

    # 2. 保存静态配置信息到TXT
    # 计算模型资源信息
    params, flops = calculate_model_resources(model, input_size=(3, 64, 64))
    training_time = (time.time() - start_time) / 3600  # 计算 GPU 训练时间（小时）
    config_path = os.path.join(results_dir, 'config', 'training_config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        # 模型信息
        f.write("===== 模型信息 =====\n")
        f.write(f"模型名称: {model.__class__.__name__}\n")
        f.write(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"优化器: {optimizer.__class__.__name__}\n")
        f.write(f"权重衰减: {optimizer.param_groups[0].get('weight_decay', 0)}\n")
        f.write(f"初始学习率: {learning_rates[0]}\n")
        f.write(f"最终学习率: {learning_rates[-1]}\n")
        f.write(f"Total FLOPs: {flops:,}\n")
        f.write(f"GPU Training Time (Hours): {training_time:.2f}h\n\n")

        # 数据集信息
        f.write("===== 数据集信息 =====\n")
        f.write(f"图像尺寸: {transform.transforms[0].size[0]}x{transform.transforms[0].size[1]}\n")
        f.write(f"训练集大小: {len(train_loader.dataset):,} 样本\n")
        f.write(f"测试集大小: {len(test_loader.dataset):,} 样本\n")
        f.write(f"验证集大小: {len(val_loader.dataset):,} 样本\n")
        f.write(
            f"总数据集大小: {len(train_loader.dataset) + len(test_loader.dataset) + len(val_loader.dataset):,} 样本\n")
        f.write(f"批次大小: {train_loader.batch_size} 样本\n")
        f.write(f"训练批次数: {len(train_loader):,}\n")
        f.write(f"测试批次数: {len(test_loader):,}\n")
        f.write(f"验证批次数: {len(val_loader):,}\n\n")

        # 训练配置
        f.write("===== 训练配置 =====\n")
        f.write(f"最大训练轮数: {num_epochs}\n")
        f.write(f"实际训练轮数: {epoch_counter}\n")
        f.write(f"早停耐心值: {patience}\n\n")

        # 训练结果摘要
        f.write("===== 训练结果 =====\n")
        f.write(f"最终训练准确率: {train_accuracies[-1]:.8f}\n")
        f.write(f"最终测试准确率: {test_accuracies[-1]:.8f}\n")
        f.write(f"最终验证准确率: {val_accuracies[-1]:.8f}\n")

    # 3. 打印训练摘要
    print("\n训练配置摘要:")
    print(f"模型: {model.__class__.__name__} (参数量: {total_params:,})")
    print(f"优化器: {optimizer.__class__.__name__}")
    print(f"实际训练轮数: {epoch_counter}/{num_epochs}")
    print(f"训练已结束，结果已保存至: {results_dir}")
    return dynamic_df
results_df = save_training_results(
    epoch_counter=num_epochs,
    train_losses=train_losses,
    test_losses=test_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    test_accuracies=test_accuracies,
    val_accuracies=val_accuracies,
    test_f1=test_f1_scores,  # 注意这里使用列表 test_f1_scores
    test_precision=test_precisions,
    test_recall=test_recalls,
    val_f1=val_f1_scores,
    val_precision=val_precisions,
    val_recall=val_recalls,
    test_improvements=test_improvements,
    val_improvements=val_improvements,
    learning_rates=learning_rates,  # 新增：学习率历史
    model=model,
    optimizer=optimizer,
    transform=transform_train,
    current_time=current_time,
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    patience=patience
)