import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import argparse
import matplotlib.pyplot as plt
from utils import create_save_dir, plot_loss, get_act_func


def my_fft(data, freq_len=40, isnorm=1):
    """
    This function performs FFT on the given data.
    Args:
        data (numpy.ndarray): The input data.
        freq_len (int): The length of the frequency.
        isnorm (int): The normalization factor.
    Returns:
        return_fft (numpy.ndarray): The FFT output array.
    """
    datat = np.squeeze(data)
    datat_fft = np.fft.fft(datat)
    ind2 = range(freq_len)
    fft_coe = datat_fft[ind2]
    if isnorm == 1:
        return_fft = np.absolute(fft_coe)
    else:
        return_fft = fft_coe
    return return_fft


def SelectPeakIndex(FFT_Data, endpoint=True):
    """
    This function selects the peak index from FFT data.
    Args:
        FFT_Data (numpy.ndarray): The FFT data array.
        endpoint (bool): Whether to include endpoints or not. Default is True.
    Returns:
        sel_ind (numpy.ndarray): Selected index array with peaks.
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


class Linear(nn.Module):
    def __init__(
        self,
        t,
        hidden_layers_width=[100],
        input_size=20,
        num_classes: int = 1000,
        act_layer: nn.Module = nn.ReLU(),
    ):
        super(Linear, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.t = t
        layers: List[nn.Module] = []
        self.layers_width = [self.input_size] + self.hidden_layers_width
        for i in range(len(self.layers_width) - 1):
            layers += [
                nn.Linear(self.layers_width[i], self.layers_width[i + 1]),
                act_layer,
            ]
        layers += [nn.Linear(self.layers_width[-1], num_classes)]
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x

    def _initialize_weights(self) -> None:
        for obj in self.modules():
            if isinstance(obj, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(
                    obj.weight.data, 0, 1 /
                    self.hidden_layers_width[0] ** (self.t)
                )
                if obj.bias is not None:
                    nn.init.normal_(
                        obj.bias.data, 0, 1 /
                        self.hidden_layers_width[0] ** (self.t)
                    )


def train_one_step(model, optimizer, loss_fn, training_input, training_target):
    """
    Train one step.

    Args:
        model (nn.Module): model.
        optimizer (optim.Optimizer): optimizer.
        loss_fn (nn.Module): loss function.

    Returns:
        tuple: tuple containing loss and outputs.
    """
    model.train()
    training_loss = 0.0
    total = 0

    data, target = training_input.to(
        device), training_target.to(device).to(torch.float)

    for _ in range(args.training_size // args.batch_size):
        optimizer.zero_grad()
        mask = np.random.choice(
            args.training_size, args.batch_size, replace=False)
        y_train = model(data[mask])
        loss = loss_fn(y_train, target[mask])
        loss.backward()
        optimizer.step()

        training_loss += loss.item() * args.batch_size
        total += args.batch_size

    outputs = model(data)
    return training_loss / total, outputs


def test(model, loss_fn, test_input, test_target):
    model.eval()
    with torch.no_grad():
        data, target = test_input.to(
            device), test_target.to(device).to(torch.float)
        outputs = model(data)
        loss = loss_fn(outputs, target)

    return loss.item(), outputs


def plot_model_output_or_target(path, training_input, training_target, test_input, output, epoch=None):
    plt.figure()
    ax = plt.gca()

    plt.plot(training_input, training_target, "b*", label="True")
    plt.plot(test_input, output, "r-", label="Test")

    ax.tick_params(labelsize=18)
    plt.legend(fontsize=18)

    pic_name = f"output{str(epoch)}.png" if epoch is not None else "target.png"
    fntmp = os.path.join(path, pic_name)
    plt.savefig(fntmp, dpi=300)
    plt.close()


def plot_freq_distr(training_target, training_output):
    """Plot frequency distribution and the heatmap of the given training target and output."""
    y_fft = my_fft(training_target) / args.training_size
    plt.figure()
    plt.semilogy(y_fft + 1e-5, label="Target")
    idx = SelectPeakIndex(y_fft, endpoint=False)
    plt.semilogy(idx, y_fft[idx] + 1e-5, "o")

    y_fft_pred = my_fft(training_output[-1]) / args.training_size
    plt.semilogy(y_fft_pred + 1e-5, label="Model output")
    plt.semilogy(idx, y_fft_pred[idx] + 1e-5, "o")

    plt.legend(fontsize=22)
    plt.xlabel("freq idx", fontsize=22)
    plt.ylabel("freq", fontsize=22)
    plt.gca().tick_params(axis="y", labelsize=22)
    plt.gca().tick_params(axis="x", labelsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(current_run_path, "fft.png"), dpi=300)
    plt.close()

    y_pred_epoch = np.squeeze(training_output)
    idx1 = idx[:3]
    abs_err = np.zeros([len(idx1), len(training_output)])
    y_fft = my_fft(training_target)
    tmp1 = y_fft[idx1]

    for i in range(len(y_pred_epoch)):
        tmp2 = my_fft(y_pred_epoch[i])[idx1]
        abs_err[:, i] = np.abs(tmp1 - tmp2) / (1e-5 + tmp1)

    plt.figure(figsize=(8, 6))
    plt.pcolor(abs_err, cmap="RdBu", vmin=0.1, vmax=1, linewidths=0.4)
    plt.colorbar()

    plt.xlabel("Epoch", fontsize=22)

    # Set the y-axis ticks and labels to 1, 2, 3, and rotate the labels
    plt.yticks([0.5, 1.5, 2.5], [1, 2, 3], rotation=0,
               fontsize=22)  # type: ignore

    # Set the y-axis tick parameters to hide the tick marks and set the tick label size
    plt.gca().yaxis.set_tick_params(size=0)
    plt.gca().tick_params(axis="y", labelsize=22)
    plt.gca().tick_params(axis="x", labelsize=22)

    plt.title("Absolute Error", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(current_run_path, "hot.png"))
    plt.close()


def get_y(x):
    y = np.sin(x) + np.sin(3 * x) + np.sin(5 * x)
    return y


def main():
    act_func = get_act_func(args.act_func_name)

    for i in range(2):
        if isinstance(args.boundary[i], str):
            args.boundary[i] = eval(args.boundary[i])

    test_input = torch.reshape(torch.linspace(
        args.boundary[0] - 0.5, args.boundary[1] + 0.5, steps=args.test_size), [args.test_size, 1])
    training_input = torch.reshape(torch.linspace(
        args.boundary[0], args.boundary[1], steps=args.training_size), [args.training_size, 1])
    test_target = get_y(test_input)
    training_target = get_y(training_input)

    plot_model_output_or_target(
        current_run_path, training_input, training_target, test_input, test_target)

    model = Linear(
        args.t, args.hidden_layers_width, args.input_dim, args.output_dim, act_func).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction="mean")

    t0 = time.time()
    loss_training_lst = []
    loss_test_lst = []
    training_output = []

    for epoch in range(args.epochs + 1):
        loss, train_output = train_one_step(
            model, optimizer, loss_fn, training_input, training_target)
        loss_test, output = test(model, loss_fn, test_input, test_target)

        loss_training_lst.append(loss)
        loss_test_lst.append(loss_test)
        training_output.append(train_output.detach().cpu().numpy())

        if epoch % args.plot_epoch == 0:
            print("[%d] loss: %.6f val_loss: %.6f time: %.2f s" %
                  (epoch + 1, loss, loss_test, (time.time() - t0)))

        if (epoch + 1) % (args.plot_epoch) == 0:
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=True)
            plot_loss(path=current_run_path,
                      loss_train=loss_training_lst, x_log=False)

            plot_model_output_or_target(
                current_run_path, training_input, training_target, test_input,
                output.detach().cpu().numpy(), epoch)

    plot_freq_distr(training_target, training_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Training for Frequency Principle")
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--optimizer", default="adam",
                        help="optimizer: sgd | adam")
    parser.add_argument("--epochs", default=1000, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch_size", default=101, type=int,
                        help="the batch size for model (default: 101)")
    parser.add_argument("--training_size", default=101, type=int,
                        help="the training data size for model (default: 101)")
    parser.add_argument("--test_size", default=1000, type=int,
                        help="the test data size for model (default: 1000)")
    parser.add_argument("--t",  type=float,  default=0.5,
                        help="parameter initialization distribution variance power(We first assume that each layer is the same width.)")
    parser.add_argument("--boundary", nargs="+",  type=str,
                        default=["-3.1415", "3.1415"], help="the boundary of 1D data")
    parser.add_argument("--act_func_name", default="tanh",
                        help="activation function: tanh | ReLU | Sigmoid | hat")
    parser.add_argument("--hidden_layers_width", nargs="+",
                        type=int, default=[200, 200, 200, 100])
    parser.add_argument("--input_dim", default=1, type=int,
                        help="the input dimension for model (default: 1)")
    parser.add_argument("--output_dim", default=1, type=int,
                        help="the output dimension for model (default: 1)")
    parser.add_argument("--plot_epoch", default=1000, type=int,
                        help="step size of plotting interval (default: 1000)")

    args, _ = parser.parse_known_args()
    current_run_path = create_save_dir(folder='1d-runs')
    print('Current run path: ', current_run_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
