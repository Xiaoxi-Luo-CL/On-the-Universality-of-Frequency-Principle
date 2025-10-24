import os
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import argparse
import yaml


def create_save_dir(folder=''):
    """
    Create a new directory with the current date and time as its name and return the path of the new directory.
    """
    subFolderName = re.sub(r'[^0-9]', '', str(datetime.datetime.now()))
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
    if act_func == 'tanh':
        return torch.nn.Tanh()
    elif act_func == 'ReLU':
        return torch.nn.ReLU()
    elif act_func == 'Sigmoid':
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
