"""Projection method of high-dimensional data, written by gemini. To be checked"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA


# --- 1D FFT 辅助函数 (来自您提供的一维代码) ---

def SelectPeakIndex(FFT_Data: np.ndarray, endpoint: bool = True) -> np.ndarray:
    """
    选择 FFT 数据中的峰值索引 (用于识别重要频率)。
    """
    D1 = FFT_Data[1:-1] - FFT_Data[0:-2]
    D2 = FFT_Data[1:-1] - FFT_Data[2:]
    D3 = np.logical_and(D1 > 0, D2 > 0)
    tmp = np.where(D3 == True)

    sel_ind = tmp[0] + 1
    if endpoint:
        if FFT_Data[0] - FFT_Data[1] > 0:
            sel_ind = np.concatenate([[0], sel_ind])

        if FFT_Data[-1] - FFT_Data[-2] > 0:
            Last_ind = len(FFT_Data) - 1
            sel_ind = np.concatenate([sel_ind, [Last_ind]])
    return sel_ind


# --- 投影法核心功能：PCA, NUDFT, Delta_F 计算 ---


def calculate_pca_projection_vector(X_flat: torch.Tensor) -> np.ndarray:
    """计算输入数据 X 的第一个主成分 p1。"""
    X_np = X_flat.cpu().numpy()
    # 标准化输入数据以进行更稳定的 PCA (尽管 PCA 内部通常会处理中心化)
    X_mean = np.mean(X_np, axis=0)
    X_centered = X_np - X_mean

    pca = PCA(n_components=1)
    pca.fit(X_centered)

    p1 = pca.components_[0]  # 第一个主成分 (单位向量)
    return p1


def project_and_sort_data(X_input: torch.Tensor, Y_target: torch.Tensor, p1: np.ndarray):
    """
    将高维数据投影到 p1 上，并根据投影值排序。
    """
    # 展平输入数据: [N, D]
    X_flat = X_input.flatten(start_dim=1).cpu().numpy()
    Y_np = Y_target.cpu().numpy()

    # 投影: x_{p,i} = x_i @ p1
    x_p = X_flat @ p1

    # 根据投影值 x_p 对数据进行排序 (NUDFT 的输入顺序)
    sort_indices = np.argsort(x_p)

    # 目标 (y_i) for NUDFT: 论文提到考虑 10-D 输出的一个维度。
    # 我们选择第一个类别/维度 (index 0)
    y_to_nudft = Y_np[sort_indices, 0]
    x_p_sorted = x_p[sort_indices]

    return x_p_sorted, y_to_nudft, sort_indices


def calculate_nudft(data_values: np.ndarray, data_locations: np.ndarray, max_freq_index: int) -> np.ndarray:
    """
    计算 Non-Uniform Discrete Fourier Transform (NUDFT) 的系数。
    \hat{f}_k = \sum_{i=0}^{N-1} f_i e^{-j \cdot 2\pi k \cdot x_i / N}

    Args:
        data_values: 排序后的函数值 f_i (即 y_i 或 h_i)。
        data_locations: 排序后的投影位置 x_i (即 x_{p,i})。
        max_freq_index: 要计算的最大频率 k。

    Returns:
        nudft_coe: 傅里叶变换复数系数 \hat{f}_k。
    """
    N = len(data_values)

    # 归一化位置到 [0, 1] 范围 (可选，但有助于 k 的物理意义)
    # 假设 data_locations 是在整个数据空间中的投影
    # 为了让 k 对应于 N 个点上的频率，我们使用 $e^{-j k x_i}$ 的形式，其中 $x_i$ 是归一化/调整后的位置
    # 在许多 NUDFT 的简化应用中，直接使用 N 作为周期

    # 为了简化，我们使用标准化的 $x_i$: $\tilde{x}_i = \frac{x_{p,i} - \min(x_{p})}{\max(x_{p}) - \min(x_{p})} \cdot (N-1)$
    x_min, x_max = data_locations.min(), data_locations.max()
    if x_max == x_min:
        x_norm = np.arange(N)  # 避免除零，使用均匀间隔作为默认
    else:
        x_norm = (data_locations - x_min) / (x_max - x_min) * (N - 1)

    nudft_coe = np.zeros(max_freq_index, dtype=np.complex128)

    for k in range(max_freq_index):
        # NUDFT: \hat{f}_k = \sum_{i=0}^{N-1} f_i e^{-j \cdot 2\pi k \cdot x_{norm, i} / N}
        term = data_values * np.exp(-1j * 2 * np.pi * k * x_norm / N)
        nudft_coe[k] = np.sum(term)

    return nudft_coe


def calculate_delta_f(target_fft: np.ndarray, output_fft: np.ndarray) -> np.ndarray:
    """计算相对频率误差 \Delta_F(k) = |\hat{h}_k − \hat{y}_k| / (|\hat{y}_k| + \epsilon)"""
    epsilon = 1e-8

    Delta_F = np.abs(output_fft - target_fft) / (np.abs(target_fft) + epsilon)
    return Delta_F


def plot_projection_heatmap(Delta_F_matrix: np.ndarray, target_magnitude: np.ndarray, path: str, max_freq: int):
    """
    绘制 Delta_F(k) 随 Epoch 变化的热力图。
    """

    plt.figure(figsize=(10, 8))

    # 裁剪数据到最大绘图频率
    plot_freq_limit = min(max_freq, Delta_F_matrix.shape[0])
    plot_data = Delta_F_matrix[:plot_freq_limit, :]

    # 绘制 \log_{10}(\Delta_F(k))
    log_plot_data = np.log10(plot_data + 1e-10)

    # vmin=-5 意味着相对误差约为 0.001%， vmax=0 意味着 100% 相对误差（未收敛）
    plt.pcolormesh(log_plot_data, cmap="RdYlGn_r",
                   vmin=-5, vmax=0, shading='nearest')
    plt.colorbar(label=r'$\log_{10}(\Delta_F(k)) = \log_{10}(|\hat{h}_k − \hat{y}_k| / (|\hat{y}_k| + \epsilon))$',
                 orientation="vertical")

    plt.xlabel("Epoch Index", fontsize=18)
    plt.ylabel(r"Frequency Index $k$ (along $\vec{p}_1$)", fontsize=18)

    # 标记重要频率 (峰值)
    peak_indices = SelectPeakIndex(
        target_magnitude[:plot_freq_limit], endpoint=False)

    for peak_k in peak_indices:
        # 在热力图上绘制水平线标记重要的频率分量
        plt.axhline(peak_k, color='w', linestyle='--',
                    linewidth=0.5, alpha=0.7)
        plt.text(Delta_F_matrix.shape[1] - 1, peak_k + 0.5, f'k={peak_k}',
                 color='w', fontsize=10, ha='right')

    plt.title(r"NUDFT $\Delta_F(k)$ 收敛热力图 (Projection Method)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "projection_heatmap_delta_f.png"), dpi=300)
    plt.close()


def run_projection_analysis(
    X_input: torch.Tensor,
    Y_target: torch.Tensor,
    H_outputs: List[np.ndarray],
    save_path: str,
    max_plot_freq: int = 50
):
    """
    主分析函数：执行 PCA、投影、NUDFT 和 Delta_F 矩阵的计算与绘图。
    """
    # 1. 计算 PC1 (p1)
    X_flat = X_input.flatten(start_dim=1)
    p1 = calculate_pca_projection_vector(X_flat)

    # 2. 投影数据、目标并排序
    x_p_sorted, y_to_nudft, sort_indices = project_and_sort_data(
        X_input, Y_target, p1)
    N = len(y_to_nudft)

    # 3. 目标标签的 NUDFT (\hat{y}_k)
    max_freq_index = N // 2
    Y_target_nudft = calculate_nudft(y_to_nudft, x_p_sorted, max_freq_index)

    # 4. 初始化误差矩阵
    num_epochs = len(H_outputs)
    num_freqs = len(Y_target_nudft)
    Delta_F_matrix = np.zeros((num_freqs, num_epochs))

    # 5. 计算每个 epoch 的 \Delta_F(k)
    print(f"--- 频率分析: N={N} 个点, 计算 {num_freqs} 个频率分量 ---")
    for i, H_epoch_np in enumerate(H_outputs):
        # 排序 DNN 输出
        # H_epoch_np 形状是 [N, num_classes=10]，选择第 0 维
        h_to_nudft = H_epoch_np[sort_indices, 0]

        # 计算 DNN 输出的 NUDFT (\hat{h}_k)
        H_output_nudft = calculate_nudft(
            h_to_nudft, x_p_sorted, max_freq_index)

        # 计算 Delta_F(k)
        Delta_F_matrix[:, i] = calculate_delta_f(
            Y_target_nudft, H_output_nudft)

        if (i+1) % 10 == 0:
            print(f"Epoch {i+1}/{num_epochs} Delta_F 计算完成。")

    print("--- 投影分析完成。开始绘图 ---")

    plot_projection_heatmap(Delta_F_matrix, np.abs(
        Y_target_nudft), save_path, max_plot_freq)


# --- 2. 训练相关函数和模型（基于用户代码，并根据论文要求修改模型） ---

class My_CNN_Project(nn.Module):
    """
    CNN model based on user's structure, modified for the paper's FC layers: 
    Conv(32, 64) -> MaxPool -> FC(12544 -> 800 -> 400 -> 400 -> 400 -> 10)
    """

    def __init__(self, in_channels=3, num_classes: int = 10, act_layer: nn.Module = nn.ReLU()):
        super(My_CNN_Project, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # 卷积层 (与用户提供的 My_CNN 结构一致)
        conv_layers: List[nn.Module] = []
        conv_layers += [nn.Conv2d(self.in_channels, 32,
                                  kernel_size=3, stride=1, padding=0), act_layer]
        conv_layers += [nn.Conv2d(32, 64, kernel_size=3,
                                  stride=1, padding=0), act_layer]
        conv_layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        self.conv = nn.Sequential(*conv_layers)

        # MLP 层 (根据论文要求修改为 800-400-400-400-10)
        # 卷积输出展平尺寸: 14*14*64 = 12544
        mlp_layers: List[nn.Module] = []

        # 12544 -> 800
        mlp_layers += [nn.Linear(14*14*64, 800), act_layer]

        # 800 -> 400
        mlp_layers += [nn.Linear(800, 400), act_layer]

        # 400 -> 400
        mlp_layers += [nn.Linear(400, 400), act_layer]

        # 400 -> 400
        mlp_layers += [nn.Linear(400, 400), act_layer]

        # 400 -> 10 (输出层)
        mlp_layers += [nn.Linear(400, self.num_classes), nn.Softmax(dim=1)]

        self.mlp = nn.Sequential(*mlp_layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.mlp(x)
        return x

    def _initialize_weights(self) -> None:
        for obj in self.modules():
            if isinstance(obj, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(obj.weight.data)


def evaluate(model, loss_fn, test_loader):
    """返回模型在测试集上的平均损失和准确率。"""
    model.eval()
    train_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            batch_size = inputs.size(0)
            total += batch_size
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # 使用 NLLLoss 需要 log(outputs)
            loss = loss_fn(torch.log(outputs), targets)
            train_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
        acc = 100 * correct / total
    return train_loss / total, acc


def val(model, training_loader):
    """返回模型在训练集上的所有预测输出 (H)。"""
    model.eval()
    y_pred = []

    with torch.no_grad():
        for inputs, targets in training_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_pred.append(outputs)

    # 返回整个训练集上的输入 X, 目标 Y, 和预测 H (为了 PCA/Projection)
    X_list = [inputs.cpu() for inputs, _ in training_loader]
    Y_list = [targets.cpu() for _, targets in training_loader]

    # X, Y_one_hot, H
    return torch.cat(X_list), F.one_hot(torch.cat(Y_list), num_classes=10), torch.cat(y_pred)


def train_one_step(model, optimizer, loss_fn, train_loader):
    """执行一个训练 epoch。"""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(torch.log(outputs), targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def load_data(training_size, training_batch_size, test_batch_size):
    """加载 CIFAR-10 数据。"""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root="./", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(
        root="./", train=False, transform=transform, download=True)

    # 限制训练集大小
    train_dataset = Subset(train_dataset, range(training_size))

    # 注意: shuffle=False 是至关重要的，这样每次 val 得到的 X, Y 顺序不变，确保 PCA 和投影的一致性。
    train_loader = DataLoader(train_dataset, batch_size=training_batch_size,
                              num_workers=1, shuffle=False,
                              drop_last=False, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    return train_loader, test_loader


def main():
    train_loader, test_loader = load_data(
        args.training_size, args.training_batch_size, args.test_batch_size
    )

    act_func = get_act_func(args.act_func_name)
    # 使用修改后的模型结构
    model = My_CNN_Project(
        args.in_channel, args.num_classes, act_func).to(device)
    # 论文提到使用 Cross-Entropy Loss, 配合模型最后的 Softmax，这里使用 NLLLoss。
    loss_fn = nn.NLLLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'nesterov':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_training_lst = []

    # 用于存储 DNN 输出历史的列表 H_outputs
    H_outputs: List[np.ndarray] = []

    t0 = time.time()

    # 第一次运行 val 获取初始 X_train 和 Y_target
    X_train_data, Y_target_one_hot, H_initial = val(model, train_loader)
    H_outputs.append(H_initial.detach().cpu().numpy())

    print(
        f"X_train shape: {X_train_data.shape}, Y_target shape: {Y_target_one_hot.shape}")

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_step(model, optimizer, loss_fn, train_loader)
        loss_test, acc_test = evaluate(model, loss_fn, test_loader)

        loss_training_lst.append(loss)

        # 获取当前 epoch 的 DNN 输出 H
        _, _, H_current = val(model, train_loader)
        H_outputs.append(H_current.detach().cpu().numpy())

        if epoch % args.plot_epoch == 0:
            print("[%d] loss: %.6f, acc: %.2f, val loss: %.6f, val acc: %.2f, time: %.2f s" %
                  (epoch, loss, acc, loss_test, acc_test, (time.time()-t0)))

        if (epoch) % (args.plot_epoch) == 0:
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=True)
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=False)

    print("\n--- 训练结束。开始运行 Projection Method 分析 ---")

    # 执行投影法分析并绘制热力图
    # Y_target_one_hot 是 [N, 10]，我们只需要其中一维 (如第0维)
    run_projection_analysis(
        X_input=X_train_data,
        Y_target=Y_target_one_hot,
        H_outputs=H_outputs,  # 包含 Epoch 0 到 Epoch N 的所有输出
        save_path=current_run_path,
        max_plot_freq=args.max_plot_freq
    )
    print("投影分析完成，热力图已保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Training for Frequency Principle (Projection Method)')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='nesterov',
                        help='optimizer: sgd | adam | nesterov')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--training_batch_size', default=100, type=int,
                        help='the batch size for model (default: 100)')
    parser.add_argument('--training_size', default=5000, type=int,
                        help='the training size for model (default: 5000)')
    parser.add_argument('--test_batch_size', default=100, type=int,
                        help='the test size for model (default: 100)')
    parser.add_argument('--act_func_name', default='ReLU',
                        help='activation function')
    parser.add_argument('--in_channel', default=3, type=int,
                        help='the input channel for model (default: 3)')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='the output dimension for model (default: 10)')
    parser.add_argument('--plot_epoch', default=10, type=int,
                        help='step size of plotting interval (default: 10)')
    parser.add_argument('--max_plot_freq', default=100, type=int,
                        help='maximum frequency index to plot on heatmap')

    args, _ = parser.parse_known_args()
    current_run_path = create_save_dir(folder='nd-projection-runs')
    print('Current run path: ', current_run_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
