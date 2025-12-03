'''Build the whole pipeline. Explorations:
1. width, number of samples, activation functions
2. Non-uniform input data (on sphere and in R^2)
3. Higher-dimensional inputs (optional)
4. switch to common MLP for exploring the effect of initialization
Written by Xiaoxi Luo, 2025 Nov 3'''

import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from decompose_NTK import NTK_empirical, plot_eigendecay, infinite_NTK_approx
from utils import create_save_dir, MLP_General, MLP
import finufft
from frequency_func import analyze_spectral_error_dynamics, analyze_filtering_dynamics
import copy


def init_inputs(num_inputs=200, dim=2, option='uniform'):
    """
    option: 'uniform' or 'sphered' or 'random'.
    'uniform' generates uniform data on (dim-1) dimensional unit sphere,
    'sphered' generates non-uniform data (dim-1) dimensional unit sphere,
    'random' generates random data in R^dim.
    """
    if dim != 2:
        raise NotImplementedError(
            "Currently only 2D input data generation is implemented.")
    if option == 'uniform':
        theta = torch.linspace(0, 2 * math.pi, num_inputs+1)[:-1]
    elif option == 'sphered':
        # Example: 80% of points in the first quadrant
        n_dense = int(num_inputs * 0.8)
        n_sparse = num_inputs - n_dense
        theta_dense = torch.rand(n_dense) * (np.pi / 2)
        theta_sparse = torch.rand(n_sparse) *\
            (2 * np.pi - np.pi / 2) + (np.pi / 2)
        theta = torch.cat([theta_dense, theta_sparse])
    elif option == 'random':  # NUFFT need data in [-pi, pi]
        x_data = torch.randn(num_inputs, dim)
        x_data[x_data > np.pi] = np.pi * 0.999
        x_data[x_data < -np.pi] = -np.pi * 0.999
        return x_data.float(), None
    elif option == 'clustered':
        print("Generating 'clustered' (GMM) data...")

        # Cluster 1: 50% of points, tight cluster in top-left
        n1 = int(num_inputs * 0.5)
        mean1 = torch.tensor([-2.0, 2.0])
        std1 = 0.3
        c1 = torch.randn(n1, dim) * std1 + mean1

        # Cluster 2: 30% of points, medium cluster in bottom-right
        n2 = int(num_inputs * 0.3)
        mean2 = torch.tensor([1.5, -1.5])
        std2 = 0.5
        c2 = torch.randn(n2, dim) * std2 + mean2

        # Cluster 3: 20% of points, wide cluster near center
        n3 = num_inputs - n1 - n2
        mean3 = torch.tensor([0.0, 0.0])
        std3 = 1.0
        c3 = torch.randn(n3, dim) * std3 + mean3

        x_data = torch.cat([c1, c2, c3], dim=0)
        x_data.clamp_(-np.pi, np.pi)
        return x_data.float(), None
    else:
        raise ValueError("Invalid option for data generation.")

    x_data = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    return x_data.float(), theta.float()


def get_target_function(x_data, theta, type=1):
    if type == 1:
        assert theta is not None, "Theta is required for target func type 1."
        return torch.sin(2*theta) + 5*torch.cos(
            5*theta) + 3*torch.sin(10*theta)
    elif type == 2:  # f(x,y) = e^(x/2) + y^2
        y1 = torch.exp(x_data[:, 0] / 2.0)
        y2 = x_data[:, 1]**2
        return (y1+y2).float()
    elif type == 3:
        # f(x, y) = \sin(2\pi x_1) + 2 e^{-5((x_1-0.5)^2 + x_2^2)}
        y1 = torch.sin(2 * np.pi * x_data[:, 0])
        y2 = 2.0 * torch.exp(-5.0 * ((x_data[:, 0] - 0.5) **
                                     2 + (x_data[:, 1] - 0.5)**2))
        return (y1 + y2).float()
    elif type == 4:
        # f(\theta) = e^{0.5 \cos\theta} + \sin^2\theta + \mathbf{0.1 \sin(15\theta)}$$
        assert theta is not None, "Theta is required for target func type 4."
        return torch.exp(0.5 * torch.cos(theta)) + torch.sin(theta)**2 + torch.sin(15 * theta)
    else:
        raise ValueError("Invalid target function type.")


def eigen_decomposition(kernel_matrix):
    """
    Performs eigen-decomposition on the kernel matrix.
    Returns eigenvalues and eigenvectors sorted by eigenvalue magnitude.
    dot(a, eigenvectors[:, i]) = eigenvalues[i] * eigenvectors[:, i]
    """
    print("Decomposing NTK matrix...")
    eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sort_indices]
    eigenvectors_sorted = eigenvectors[:, sort_indices]
    return eigenvalues_sorted, eigenvectors_sorted


def eigenvector_spectrum(eigenvectors, x_data, vec_list_to_plot, theta,
                         fig_name="eigenvector_spectrum"):
    """
    Analyzes and plots the Fourier spectrum of the given eigenvectors,
    auto-selecting 1D-FFT, 1D-NUFFT, 2D-NUFFT based on the data.
    TODO: 3D-NUFFT can be added similarly.
    """
    print("Analyzing spectrum of NTK eigenvectors...")
    assert vec_list_to_plot, "No eigenvectors specified for analysis!"

    for vec_idx in vec_list_to_plot:
        v_k = np.real(eigenvectors[:, vec_idx])

        # Case 1: Uniform data on sphere, theta is uniform
        if DATA_OPTION == 'uniform':
            fft_vals = np.fft.fft(v_k)
            num_freq = min(len(theta)//2, 50)
            power = np.abs(fft_vals[0:num_freq])**2
            freqs = np.arange(num_freq)

            plt.plot(freqs, power, 'b-o', markersize=4)
            plt.title(f"Spectrum of Eigvec {vec_idx} (1D FFT)")
            plt.ylabel("Power")
            plt.xlabel("Frequency (k)")

        # Case 2: 2D Non-Uniform data on Circle (theta is non-uniform)
        # finufft1d1: (theta, c_j) -> f_k
        # theta needs to be rescaled to [-pi, pi], c_j are values
        elif DATA_OPTION == 'sphered':
            n_freqs = 51  # check first 50 frequencies
            theta_np = theta.numpy() - np.pi
            f_k = finufft.nufft1d1(
                theta_np, v_k.astype(np.complex64), n_freqs)
            freqs_all = np.arange(-25, 26)   # Freqs -25~25
            power = np.abs(f_k)**2

            plt.plot(freqs_all, power, 'r-o', markersize=4)
            plt.title(f"Spectrum of Eigenvector {vec_idx} (1D NUFFT)")
            plt.ylabel("Power")
            plt.xlabel("Frequency (k)")

        # Case 3: 2D Non-Uniform data in R^2 (theta is None).
        # finufft2d1: (x_j, y_j, c_j) -> f_k
        else:
            n_freqs = 21
            x_scaled = np.ascontiguousarray(x_data[:, 0].numpy())
            y_scaled = np.ascontiguousarray(x_data[:, 1].numpy())
            f_k = finufft.nufft2d1(x_scaled, y_scaled, v_k.astype(
                np.complex64), (n_freqs, n_freqs), isign=-1)
            power_2d = np.abs(f_k)**2

            plt.imshow(power_2d, origin='lower',
                       extent=(-10, 10, -10, 10))
            plt.title(f"Spectrum of Eigenvector {vec_idx} (2D NUFFT)")
            plt.xlabel("Frequency k_x")
            plt.ylabel("Frequency k_y")
            plt.colorbar(label="Power")

        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/{fig_name}-eigv{vec_idx}.png')
        plt.clf()


def train(model, x_data, y_target, ntk_eigenvectors, ntk_eigenvalues,
          indices_to_track: list, fig_name: str, lr=1e-3, steps=10000):
    """
    Trains the model and plots the decay of the residual projected onto the NTK eigenvectors.
    """
    print("Running training dynamics experiment...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    projection_history = {k: [] for k in indices_to_track}
    step_history = []
    loss_history = []
    y_pred_history = []
    v_k_tensors = {k: torch.Tensor(
        np.real(ntk_eigenvectors[:, k])) for k in indices_to_track}

    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        y_pred = model(x_data)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % LOG_INTERVAL == 0:
            print(f'step {step}, loss = ', loss.item())
            step_history.append(step)
            y_pred_history.append(y_pred.detach().numpy())
            with torch.no_grad():
                residual_vec = (y_target - y_pred).detach().squeeze()
                for k in indices_to_track:
                    v_k = v_k_tensors[k]
                    proj_len = torch.abs(torch.dot(residual_vec, v_k))
                    projection_history[k].append(proj_len.item())

    # Plot the projection decay
    plt.figure(figsize=(8, 6))
    for k in indices_to_track:
        plt.plot(step_history, projection_history[k],
                 label=f"Eigvec {k} (λ={ntk_eigenvalues[k]:.2e})")
    plt.xlabel("Training Step", fontsize=18)
    plt.ylabel("Projection Length of Residual", fontsize=18)
    # plt.title("Training Dynamics Projected onto NTK Eigenbasis")
    plt.yscale('log')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{fig_name}.png')
    plt.clf()

    # plot training loss
    plt.plot(loss_history, 'k-')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig(f'{SAVE_DIR}/loss{fig_name}.jpg')
    plt.clf()
    return projection_history, step_history, model, y_pred_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_input', type=int, default=2,
                        help='Input dimension')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of data points')
    parser.add_argument('--width', type=int, default=1000,
                        help='Width of the MLP')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="Activation function: 'ReLU' or 'Tanh'")
    parser.add_argument('--data_option', type=str, default='uniform',
                        help="Data generation option: 'uniform', 'sphered', 'random', 'clustered'")
    parser.add_argument('--target_func', type=int,
                        default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--steps', type=int, default=20000,
                        help="Training steps")
    parser.add_argument('--initialization', type=str, default='ntk',
                        choices=['kaiming', 'ntk', 'mf'], help="Initialization method")
    args = parser.parse_args()

    # --- Setting hyperparameters ---
    INPUT_DIM = args.d_input
    NUM_SAMPLES = args.n_samples
    WIDTH = args.width
    DATA_OPTION = args.data_option
    TARGET_OPTION = args.target_func
    ACTIVATION = args.activation
    LEARNING_RATE = args.lr
    TRAIN_STEPS = args.steps
    LOG_INTERVAL = 200
    INITIAL = args.initialization
    file_name = f'_n{NUM_SAMPLES}_w{WIDTH}_{DATA_OPTION}_f{TARGET_OPTION}_{ACTIVATION}_steps{TRAIN_STEPS}_lr{LEARNING_RATE}_{INITIAL}'

    torch.manual_seed(42)
    np.random.seed(42)
    mlp_for_ntk = MLP_General(input_dim=INPUT_DIM, hidden_dim=WIDTH,
                              activation=ACTIVATION, parameterization=INITIAL)
    params_init = copy.deepcopy(dict(mlp_for_ntk.named_parameters()))

    # define data and model
    x_data, theta = init_inputs(
        num_inputs=NUM_SAMPLES, dim=INPUT_DIM, option=DATA_OPTION)

    SAVE_DIR = create_save_dir('experiments/pipeline', suffix=file_name)
    print('Current run path: ', SAVE_DIR)

    # --- A. static NTK analysis ---
    empirical_kernel = NTK_empirical(mlp_for_ntk, x_data)
    eigenvalues, eigenvectors = eigen_decomposition(empirical_kernel)
    # vec_list = list(range(10)) + [i*10 for i in range(1, 6)]
    # plot_eigendecay([empirical_kernel], [f'Empirical {ACTIVATION}'],
    #                 fig_name=f'eigendecay_w{WIDTH}_n{NUM_SAMPLES}_{ACTIVATION}',
    #                 fig_path=f'{SAVE_DIR}/eigendecay_w{WIDTH}_n{NUM_SAMPLES}_{ACTIVATION}',
    #                 loglog=True, eigen_num=100)
    # eigenvector_spectrum(eigenvectors, x_data,
    #                      vec_list_to_plot=vec_list, theta=theta,
    #                      fig_name=f"spectrum_{DATA_OPTION}_{ACTIVATION}")

    # --- B. dynamic training analysis ---
    y_target = get_target_function(
        x_data, theta, TARGET_OPTION).unsqueeze(1)

    proj_hist, steps, trained_model, y_pred_hist = train(
        mlp_for_ntk, x_data, y_target,
        eigenvectors, eigenvalues, indices_to_track=[0, 1, 2, 3, 4, 10, 20],
        fig_name=f"train{file_name}",
        lr=LEARNING_RATE, steps=TRAIN_STEPS
    )

    analyze_spectral_error_dynamics(
        x_data, y_target, y_pred_hist, theta, steps, DATA_OPTION,
        SAVE_DIR, file_name)
    analyze_filtering_dynamics(
        x_data.numpy(), y_target.numpy(), y_pred_hist, SAVE_DIR)

    params_final = dict(trained_model.named_parameters())

    relative_changes = {}

    for name in params_init:
        w0 = params_init[name]
        w1 = params_final[name].detach()
        diff = torch.norm(w1 - w0)
        base = torch.norm(w0)
        relative_changes[name] = (diff / (base + 1e-10)).item()

    for k, v in relative_changes.items():
        print(f"{k}: relative change = {v:.6e}")
