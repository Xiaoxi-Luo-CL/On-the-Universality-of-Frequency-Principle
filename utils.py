import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import argparse
import yaml
import math
import torch.nn.init as init
import torch.nn as nn


class MLP(nn.Module):
    """A simple two-layer MLP with ReLU activation."""

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation='relu'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = get_act_func(activation)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MLP_NTK(nn.Module):
    """
    A two-layer MLP with NTK initialization.
    This ensures that the empirical NTK scales as O(1) with respect to width.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, activation: str = 'relu'):
        super(MLP_NTK, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = get_act_func(activation)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self._ntk_init(input_dim=input_dim)

    def _ntk_init(self, input_dim: int):
        """
        Applies strict Gaussian initialization based on the NTK/Kernel parameterization.
        Variance is set to 1/fan_in for all layers. Biases are zero.
        """
        std_w1 = 1.0 / math.sqrt(input_dim)
        init.normal_(self.layer1.weight, mean=0.0, std=std_w1)
        if self.layer1.bias is not None:
            init.zeros_(self.layer1.bias)

        init.normal_(self.layer2.weight, mean=0.0, std=1)
        if self.layer2.bias is not None:
            init.zeros_(self.layer2.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x / math.sqrt(self.hidden_dim)


def create_save_dir(folder='', suffix=''):
    """
    Create a new directory with the current date and time as its name and return the path of the new directory.
    """
    subFolderName = re.sub(
        r'[^0-9]', '', str(datetime.datetime.now()))[:10]+suffix
    path = os.path.join(folder, subFolderName)
    os.makedirs(path, exist_ok=True)
    # my_mkdir(os.path.join(path, 'output'))
    return path


def plot_diff_distr(filter, lowdiff, highdiff, current_run_path):
    """
    Plot the difference between low and high frequency components of predictions vs. targets.

    Args:
        filter (float): The filter value used for frequency separation.
        lowdiff (list/np.array): Relative distances for the low frequency component, typically 
                                 tracked across epochs.
        highdiff (list/np.array): Relative distances for the high frequency component, typically 
                                  tracked across epochs.
    """
    lowdiff = np.array(lowdiff)
    highdiff = np.array(highdiff)
    num_epochs = len(lowdiff)
    epochs = np.arange(1, num_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Relative Error with Filter {:0.2f}'.format(filter))

    # --- Left Subplot: Line Plot ---
    ax0 = axes[0]
    # Plot low-frequency differences against epochs
    ax0.plot(epochs, lowdiff, 'r-', label='low_{:0.2f}'.format(filter))
    # Plot high-frequency differences against epochs
    ax0.plot(epochs, highdiff, 'b-', label='high_{:0.2f}'.format(filter))
    ax0.legend()
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('error')
    ax0.set_xticks(np.arange(0, num_epochs + 1, 20))

    # --- Right Subplot: Heatmap ---
    ax1 = axes[1]
    tmp = np.stack([lowdiff, highdiff])

    # 2. Use pcolor, explicitly passing X and Y coordinates
    pc = ax1.pcolor(epochs, [0.5, 1.5], tmp, cmap='RdBu', vmin=0.1, vmax=1)
    ax1.set_xticks(np.arange(0, num_epochs + 1, 20))

    # 3. Fix Y-axis labels for the heatmap
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(('low freq', 'high freq'), rotation='vertical')
    ax1.set_xlabel('epoch')

    fig.colorbar(pc, ax=ax1, ticks=np.arange(0.1, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(current_run_path + '/hot_{:0.2f}.png'.format(filter))
    plt.close()


def plot_loss(path, loss_train, x_log=False):
    """
    path (str): path.
    loss_train (list): list of training loss.
    x_log (bool): whether to use log scale for x-axis.
    """
    plt.figure()
    ax = plt.gca()
    y2 = np.asarray(loss_train)
    plt.plot(y2, 'k-', label='Train')
    plt.xlabel('epoch', fontsize=18)
    ax.tick_params(labelsize=18)
    plt.yscale('log')
    if x_log == False:
        path = os.path.join(path, 'loss.jpg')
    else:
        plt.xscale('log')
        path = os.path.join(path, 'loss_log.jpg')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def get_act_func(act_func):
    if act_func.lower() == 'tanh':
        return torch.nn.Tanh()
    elif act_func.lower() == 'relu':
        return torch.nn.ReLU()
    elif act_func.lower() == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        raise NameError('No such act func!')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config():
    parser = argparse.ArgumentParser(
        description="Spectral Bias NLP Experiment")
    parser.add_argument(
        "--config", type=str, default="nlp-config.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--sample_docs", type=int,
                        help="Override number of sampled documents")
    parser.add_argument("--ngrams_per_doc", type=int,
                        help="Override n-grams per document")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.sample_docs is not None:
        cfg["dataset"]["num_docs"] = args.sample_docs
    if args.ngrams_per_doc is not None:
        cfg["dataset"]["ngrams_per_doc"] = args.ngrams_per_doc

    return cfg


def load_spectral_npz(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")],
                   key=lambda x: int(x.split("_")[-1].split(".npz")[0]))
    lows, highs = [], []
    for f in files:
        data = np.load(os.path.join(folder, f))
        lows.append(data['low'])
        highs.append(data['high'])

    return np.stack(lows), np.stack(highs)


def calculate_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()
    return norm


def relu(x):
    return np.maximum(x, 0)


def d_relu(x):
    return 1.0 * (x > 0)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def d_cos(x):
    return -np.sin(x)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1.0 - np.tanh(x)**2
