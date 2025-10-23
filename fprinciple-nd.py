from typing import Tuple
from torch.utils.data import DataLoader, Subset
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import random
from utils import create_save_dir, plot_diff_distr, plot_loss, get_act_func


def compute_distance_for_training_loader(training_loader, sampled=True):
    """
    Computes the pairwise Euclidean distance between all pairs of flattened images in a training loader.

    Args:
        training_loader (DataLoader): A DataLoader object containing the training data.

    Returns:
        np.ndarray: A 2D NumPy array of shape [N, N], where N is the number of images in the training set.
    """
    data_list = [data for data, _ in training_loader]

    if not sampled:
        # data = torch.cat(data_list)
        data = np.concatenate(data_list)
    else:
        data = np.array(data_list)
    # Reshape data into [B, C*H*W]
    flattened_images = data.reshape(data.shape[0], -1)

    dist = -2 * np.dot(flattened_images, flattened_images.T) + np.sum(flattened_images**2, axis=1) + \
        np.sum(flattened_images**2, axis=1)[:, np.newaxis]
    return dist


def normal_kernel(dist, filter_dict):
    """
    Computes the normalized Gaussian kernel for each filter in the filter dictionary.

    Args:
        dist (np.ndarray): A 2D NumPy array of pairwise distances between data points.
        filter_dict (list): A list of filter values to use for each kernel.

    Returns:
        list: A list of 2D NumPy arrays, where each array is a normalized Gaussian kernel for a filter in the filter dictionary.
    """
    kernel_dict = []  # list of (5k, 5k)
    for filter in filter_dict:
        kernel = np.exp(-dist / 2 / filter)
        mean = np.sum(kernel, axis=1, keepdims=True)
        kernel_dict.append(kernel/mean)
    return kernel_dict


def gauss_filiter(f_orig, kernel):
    """Applies a Gaussian filter to an image output."""
    return np.matmul(kernel, f_orig)


def get_freq_low_high(yy, kernel_dict):
    """
    Computes the low and high frequency components of the model output using a set of Gaussian filters.

    Args:
        yy (np.ndarray): NumPy array representing the model output.
        kernel_dict (list): A list of 2D NumPy arrays (eg: 5000 * 5000) representing the Gaussian kernels to use for filtering.

    Returns:
        tuple: A tuple of two lists, where the first list contains the low frequency components of
                the model output and the second list contains the high frequency components of the model output.
    """
    f_low, f_high = [], []
    for filter in range(len(kernel_dict)):
        kernel = kernel_dict[filter]
        f_new_norm = gauss_filiter(yy, kernel)  # (5000, 10)
        f_low.append(f_new_norm)
        f_high_tmp = yy - f_new_norm  # (5000, 10)
        f_high.append(f_high_tmp)
    return f_low, f_high


def get_target_freq_distr(train_labels, dist, filter_start, filter_end, filter_num):
    """
    Computes the target frequency distribution of a set of training labels using a set of Gaussian filters.

    Args:
        train_labels (np.ndarray): NumPy array representing the training labels.
        dist (np.ndarray): A 2D NumPy array of pairwise Euclidean distance between all pairs of flattened images in a training loader.
        filter_start (float): The starting value of the filter range.
        filter_end (float): The ending value of the filter range.
        filter_num (int): The number of filters to use in the filter range.

    Returns:
        tuple: A tuple of four elements, where the first element is a 1D NumPy array of filter values,
                the second element is a list of 2D NumPy arrays representing the Gaussian kernels for filtering,
                the third element is a list (length 20) of 1D NumPy arrays (5000, 10) representing the low frequency components of the training labels,
                and the fourth element is a list of 1D NumPy arrays representing the high frequency components of the training labels.
    """
    filter_dict = np.linspace(filter_start, filter_end, num=filter_num)
    kernel_dict = normal_kernel(dist, filter_dict)
    f_low, f_high = get_freq_low_high(train_labels, kernel_dict)
    return filter_dict, kernel_dict, f_low, f_high


def evaluate(model, loss_fn, test_loader):
    """
    Returns the average loss and accuracy of the model on the test set.

    :param model: the model
    :param loss_fn: the loss function
    :param test_loader: a DataLoader object
    :param args: a dictionary containing all the parameters for the training process
    :return: The loss and accuracy of the model on the test set.
    """
    model.eval()
    train_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            batch_size = inputs.size(0)
            total += batch_size
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(torch.log(outputs), targets)
            train_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
        acc = 100 * correct / total
    return train_loss / total, acc


def val(model, training_loader):
    """
    It takes a model, a test_loader, a loss function, and some arguments, and returns the average loss
    and accuracy of the model on the test set.

    :param model: the model
    :param training_loader: a DataLoader object
    :param args: a dictionary containing all the parameters for the training process
    :return: the outputs of the model on the training set
    """
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in training_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred.append(outputs)
    return torch.cat(y_pred)


def load_data(training_size, training_batch_size, test_batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root="./", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(
        root="./", train=False, transform=transform, download=True)
    train_dataset = Subset(train_dataset, range(training_size))

    train_loader = DataLoader(train_dataset, batch_size=training_batch_size,
                              num_workers=1, shuffle=True,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                             shuffle=True, num_workers=1,
                             drop_last=True, pin_memory=True)
    return train_dataset, train_loader, test_loader


class My_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes: int = 10, act_layer: nn.Module = nn.ReLU()):
        super(My_CNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        conv_layers: List[nn.Module] = []

        conv_layers += [nn.Conv2d(self.in_channels,
                                  32, kernel_size=3, stride=1, padding=0), act_layer]
        conv_layers += [nn.Conv2d(32, 64, kernel_size=3,
                                  stride=1, padding=0), act_layer]
        conv_layers += [nn.MaxPool2d(kernel_size=(2, 2))]
        self.conv = nn.Sequential(*conv_layers)
        mlp_layers: List[nn.Module] = []
        mlp_layers += [nn.Linear(14*14*64, 400), act_layer]
        mlp_layers += [nn.Linear(400, self.num_classes), nn.Softmax(dim=1)]
        self.mlp = nn.Sequential(*mlp_layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

    def _initialize_weights(self) -> None:
        for obj in self.modules():
            if isinstance(obj, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(obj.weight.data)


def train_one_step(model, optimizer, loss_fn, train_loader):
    """
    Perform one training epoch on the given DataLoader.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        loss_fn (callable): Loss function (e.g. nn.NLLLoss()).
        train_loader (DataLoader): DataLoader for training data.
        args (argparse.Namespace): Arguments containing at least 'device'.

    Returns:
        tuple[float, float]: (average training loss, training accuracy)
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(torch.log(outputs), targets)
        loss.backward()
        optimizer.step()

        # accumulate loss and accuracy
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def main_sampled():
    train_dataset, train_loader, test_loader = load_data(
        args.training_size, args.training_batch_size, args.test_batch_size
    )
    train_loader, test_loader = list(train_loader), list(test_loader)
    targets = torch.Tensor([tgt for _, tgt in train_dataset]).to(torch.int64)
    train_labels = F.one_hot(targets, num_classes=10).detach().cpu().numpy()

    # compute distance matrix for sampled set
    sample_idx = np.random.choice(len(train_dataset), 2000)
    # sample_idx = np.arange(5000)
    sampled_train = Subset(train_dataset, sample_idx.tolist())
    dist = compute_distance_for_training_loader(sampled_train)

    # target (ground-truth) frequency distribution
    filter_dict, kernel_dict, f_low, f_high = get_target_freq_distr(
        train_labels[sample_idx], dist, args.start_filter, args.end_filter, args.num_filter)

    act_func = get_act_func(args.act_func_name)
    model = My_CNN(args.in_channel, args.num_classes, act_func).to(device)
    loss_fn = nn.NLLLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'nesterov':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lowdiff = [[] for _ in filter_dict]
    highdiff = [[] for _ in filter_dict]

    t0 = time.time()
    loss_training_lst = []

    for epoch in range(args.epochs+1):
        loss, acc = train_one_step(model, optimizer, loss_fn, train_loader)
        loss_test, acc_test = evaluate(model, loss_fn, test_loader)

        # all the predictions on training set
        sampled_loader = DataLoader(sampled_train, batch_size=args.training_batch_size,
                                    num_workers=1, shuffle=False, drop_last=True, pin_memory=True)
        y_pred = val(model, sampled_loader)

        f_train_low, f_train_high = get_freq_low_high(
            y_pred.detach().cpu().numpy(), kernel_dict)

        for i in range(len(filter_dict)):
            lowdiff[i].append(np.linalg.norm(
                f_train_low[i] - f_low[i])/np.linalg.norm(f_low[i]))
            highdiff[i].append(np.linalg.norm(
                f_train_high[i] - f_high[i])/np.linalg.norm(f_high[i]))

        loss_training_lst.append(loss)
        if epoch % args.plot_epoch == 0:
            print("[%d] loss: %.6f, acc: %.2f, val loss: %.6f, val acc: %.2f, time: %.2f s" %
                  (epoch + 1, loss, acc, loss_test, acc_test, (time.time()-t0)))

        if (epoch+1) % (args.plot_epoch) == 0:
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=True)
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=False)

    for filter_index, filter in enumerate(filter_dict):
        lowdiff_ind, highdiff_ind = lowdiff[filter_index], highdiff[filter_index]
        plot_diff_distr(filter, lowdiff_ind, highdiff_ind, current_run_path)


def main():
    _, train_loader, test_loader = load_data(
        args.training_size, args.training_batch_size, args.test_batch_size
    )
    train_loader, test_loader = list(train_loader), list(test_loader)

    target_list = [target for _, target in train_loader]
    targets = torch.cat(target_list)
    train_labels = F.one_hot(targets, num_classes=10).detach().cpu().numpy()

    dist = compute_distance_for_training_loader(train_loader, sampled=False)
    filter_dict, kernel_dict, f_low, f_high = get_target_freq_distr(
        train_labels, dist, args.start_filter, args.end_filter, args.num_filter)

    act_func = get_act_func(args.act_func_name)
    model = My_CNN(args.in_channel, args.num_classes, act_func).to(device)
    loss_fn = nn.NLLLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'nesterov':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lowdiff = [[] for _ in filter_dict]
    highdiff = [[] for _ in filter_dict]

    t0 = time.time()
    loss_training_lst = []

    for epoch in range(args.epochs+1):
        loss, acc = train_one_step(model, optimizer, loss_fn, train_loader)
        loss_test, acc_test = evaluate(model, loss_fn, test_loader)

        # all the predictions on training set
        y_pred = val(model, train_loader)
        f_train_low, f_train_high = get_freq_low_high(
            y_pred.detach().cpu().numpy(), kernel_dict)

        for i in range(len(filter_dict)):
            lowdiff[i].append(np.linalg.norm(
                f_train_low[i] - f_low[i])/np.linalg.norm(f_low[i]))
            highdiff[i].append(np.linalg.norm(
                f_train_high[i] - f_high[i])/np.linalg.norm(f_high[i]))

        loss_training_lst.append(loss)

        if epoch % args.plot_epoch == 0:
            print("[%d] loss: %.6f, acc: %.2f, val loss: %.6f, val acc: %.2f, time: %.2f s" %
                  (epoch + 1, loss, acc, loss_test, acc_test, (time.time()-t0)))

        if (epoch+1) % (args.plot_epoch) == 0:
            print(len(loss_training_lst))
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=True)
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=False)

    for filter_index, filter in enumerate(filter_dict):
        lowdiff_ind, highdiff_ind = lowdiff[filter_index], highdiff[filter_index]
        plot_diff_distr(filter, lowdiff_ind, highdiff_ind, current_run_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PyTorch Training for Frequency Principle')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='nesterov',
                        help='optimizer: sgd | adam | nesterov')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--training_batch_size', default=50, type=int,
                        help='the batch size for model')
    parser.add_argument('--training_size', default=5000, type=int,
                        help='the training size for model')
    parser.add_argument('--test_batch_size', default=50, type=int,
                        help='the test size for model')
    parser.add_argument('--act_func_name', default='ReLU',
                        help='activation function')
    parser.add_argument('--in_channel', default=3, type=int,
                        help='the input channel for model (default: 3)')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='the output dimension for model (default: 10)')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device used to train (cpu or cuda)')
    parser.add_argument('--plot_epoch', default=1, type=int,
                        help='step size of plotting interval (default: 1000)')
    parser.add_argument('--ini_path', type=str,
                        default='')
    parser.add_argument('--start_filter', default=2, type=float,
                        help='the start point of the filter (default: 2)')
    parser.add_argument('--end_filter', default=100, type=float,
                        help='the end point of the filter (default: 100)')
    parser.add_argument('--num_filter', default=20, type=int,
                        help='the point number of the filter (default: 20)')

    args, _ = parser.parse_known_args()
    current_run_path = create_save_dir(folder='nd-runs')
    print('Current run path: ', current_run_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main_sampled()
