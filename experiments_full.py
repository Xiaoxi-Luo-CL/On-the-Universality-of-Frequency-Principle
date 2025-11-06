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
from decompose_NTK import MLP_NTK, NTK_empirical, plot_eigendecay, NTK_analytical_relu
from utils import create_save_dir
import finufft
import os
from scipy.stats import linregress


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
        # Example: 80% of points in the first quadrant for d=2
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


def _analyze_spectrum_core(ax, y_values, x_data, theta, title):
    """
    Analyzes and plots the Fourier spectrum of the given eigenvectors,
    auto-selecting 1D-FFT, 1D-NUFFT, 2D-NUFFT based on the data.
    TODO: 3D-NUFFT can be added similarly.
    """
    # Case 1: 2D Uniform data on Circle (theta is not None and uniform)
    if DATA_OPTION == 'uniform':
        fft_vals = np.fft.fft(y_values)
        num_freq = min(len(theta)//2, 50)
        power = np.abs(fft_vals[0:num_freq])**2
        freqs = np.arange(num_freq)

        ax.plot(freqs, power, 'b-o', markersize=4)
        ax.set_title(f"{title} (1D Uniform FFT)")
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency (k)")

    # Case 2: 2D Non-Uniform data on Circle (theta is non-uniform)
    elif DATA_OPTION == 'sphered':
        n_freqs = 51
        theta_np = theta.numpy() - np.pi
        f_k = finufft.nufft1d1(
            theta_np, y_values.astype(np.complex64), n_freqs)
        freqs = np.arange(-25, 26)   # Freqs -25~25
        power = np.abs(f_k)**2

        ax.plot(freqs, power, 'b-o', markersize=4)
        ax.set_title(f"{title} (1D NUFFT)")
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency (k)")

    # Case 3: 2D Non-Uniform data in R^2 (theta is None).
    else:
        n_freqs = 21
        x_scaled = x_data[:, 0].numpy()
        y_scaled = x_data[:, 1].numpy()

        f_k = finufft.nufft2d1(x_scaled, y_scaled, y_values.astype(
            np.complex64), (n_freqs, n_freqs))
        power_2d = np.abs(f_k)**2

        im = ax.imshow(power_2d, origin='lower',
                       extent=(-10, 10, -10, 10))
        ax.set_title(f"{title} (2D NUFFT)")
        ax.set_xlabel("Frequency k_x")
        ax.set_ylabel("Frequency k_y")
        plt.colorbar(im, ax=ax, label="Power")


def eigenvector_spectrum(eigenvectors, x_data, vec_list_to_plot, theta, fig_name="eigenvector_spectrum"):
    """
    Draw and analyze spectrum of NTK eigenvectors.
    """
    assert vec_list_to_plot, "No eigenvectors specified for analysis!"

    for vec_idx in vec_list_to_plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        v_k = np.real(eigenvectors[:, vec_idx])
        _analyze_spectrum_core(ax, v_k, x_data, theta,
                               title=f"Spectrum of Eigenvector {vec_idx}")
        plt.tight_layout()
        plt.savefig(f'{SAVE_DIR}/{fig_name}-eigv{vec_idx}.png')
        plt.close(fig)
        plt.clf()


def function_spectrum(y_func, x_data, theta, title, fig_name="function_spectrum.png"):
    """
    Draw a function's (eg: y_target or y_pred) Fourier spectrum.
    """
    print(f"Analyzing spectrum of function: {title}")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    y_values = y_func.squeeze().detach().numpy()
    _analyze_spectrum_core(ax, y_values, x_data, theta, title=title)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{fig_name}')
    plt.close(fig)
    plt.clf()


def train(model, x_data, y_target, ntk_eigenvectors, ntk_eigenvalues,
          indices_to_track: list, fig_name: str, lr=1e-3, steps=10000):
    """train the model and track projection of residual onto selected NTK eigenvectors."""
    print("Training ...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    projection_history = {k: [] for k in indices_to_track}
    step_history = []
    loss_history = []
    v_k_tensors = {k: torch.from_numpy(
        np.real(ntk_eigenvectors[:, k])).float() for k in indices_to_track}

    for step in tqdm(range(steps)):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_data)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % LOG_INTERVAL == 0:
            print(f'step {step}, loss = ', loss.item())
            step_history.append(step)
            with torch.no_grad():
                residual_vec = (y_target - y_pred).detach().squeeze()
                for k in indices_to_track:
                    v_k = v_k_tensors[k]
                    proj_len = torch.abs(torch.dot(residual_vec, v_k))
                    projection_history[k].append(proj_len.item())

    # Plot the projection decay
    plt.figure(figsize=(10, 7))
    for k in indices_to_track:
        plt.plot(step_history, projection_history[k],
                 label=f"Eigvec {k} (λ={ntk_eigenvalues[k]:.2e})")
    plt.xlabel("Training Step")
    plt.ylabel("Projection Length of Residual")
    plt.title("Training Residual Projected onto NTK Eigenbasis")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--')
    plt.savefig(f'{SAVE_DIR}/{fig_name}.png')
    plt.clf()

    # plot training loss
    plt.figure(figsize=(10, 7))
    plt.plot(loss_history, 'k-')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title("Training Loss Curve")
    plt.grid(True, which="both", linestyle='--')
    plt.savefig(f'{SAVE_DIR}/loss_{fig_name}')
    plt.clf()

    return projection_history, step_history, model


def analyze_decay_rate_vs_eigenvalue(projection_history, step_history, ntk_eigenvalues,
                                     indices_to_track, fig_name):
    """Do linear regression to analyze decay rate vs. eigenvalue."""
    decay_rates = []
    eigvals_list = []

    x_steps = np.array(step_history)

    for k in indices_to_track:
        proj_hist = np.array(projection_history[k])
        # To avoid log(0) errors, only fit when projection length > 1e-9, and only use the "stable" phase of training (e.g., first 75%)
        fit_steps = int(len(x_steps) * 0.75)
        if fit_steps < 2:
            print(f"Skipping Eigenvector {k}: Not enough data points to fit.")
            continue

        x_fit = x_steps[:fit_steps]
        y_log_proj = np.log(proj_hist[:fit_steps] + 1e-10)

        # Linear regression: log(y) = slope * x + intercept
        try:
            res = linregress(x_fit, y_log_proj)
            slope = res.slope
            r_value = res.rvalue
        except ValueError:
            print(f"Skipping Eigenvector {k}: Linear regression failed.")
            continue

        if slope < 0:
            decay_rates.append(-slope)
            eigvals_list.append(ntk_eigenvalues[k])
            print(
                f"sEigenvector {k}: λ={ntk_eigenvalues[k]:.2e}, Decay Rate (slope)={-slope:.2e}, R^2={r_value**2:.3f}")
        else:
            print(
                f"Skipping Eigenvector {k}: Non-negative slope ({slope:.2e}), training may be unstable.")

    if not eigvals_list:
        print("No valid decay rates found to plot.")
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(eigvals_list, decay_rates, alpha=0.7, c='blue')

    # Hope the correlation is y = a*x
    try:
        fit_A = np.array(eigvals_list)
        fit_b = np.array(decay_rates)
        slope, _, _, _ = np.linalg.lstsq(
            fit_A[:, np.newaxis], fit_b, rcond=None)
        slope = slope

        if slope > 0:
            x_line = np.linspace(min(eigvals_list), max(eigvals_list), 100)
            plt.plot(x_line, slope * x_line, 'r--',
                     label=f'Fit: rate ≈ {slope:.2e} * λ')
            print(
                f"Linear fit (no intercept): Decay Rate ≈ {slope:.2e} * Eigenvalue")
    except Exception as e:
        print(f"Could not fit linear trend: {e}")

    plt.xlabel("NTK Eigenvalue (λ_k)")
    plt.ylabel("Empirical Decay Rate (from log-linear fit)")
    plt.title("Learning Rate vs. NTK Eigenvalue")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--')
    plt.savefig(os.path.join(SAVE_DIR, fig_name))
    plt.clf()


def full_pipeline(kernel, vec_list, label='ANA'):
    '''Full pipeline analysis for given NTK kernel (analytical or empirical).'''
    assert label in ['ANA', 'EMP'], "Type must be 'ANA' or 'EMP'"
    eigenvalues, eigenvectors = eigen_decomposition(kernel)
    plot_eigendecay([kernel], [f'{label} NTK'],
                    sorted_eigenval=[eigenvalues],
                    fig_name=f'{label}_eigendecay_w{WIDTH}_n{NUM_SAMPLES}',
                    fig_path=f'{SAVE_DIR}/{label}_eigendecay_w{WIDTH}_n{NUM_SAMPLES}',
                    loglog=False, eigen_num=min(100, NUM_SAMPLES))
    eigenvector_spectrum(eigenvectors, x_data, vec_list, theta,
                         fig_name=f"{label}_spectrum_{DATA_OPTION}")

    # --- training dynamics ---
    torch.manual_seed(42)
    np.random.seed(42)

    mlp_to_train = MLP_NTK(
        input_dim=INPUT_DIM, hidden_dim=WIDTH, activation=ACTIVATION)

    hist, steps, trained_model = train(
        mlp_to_train, x_data, y_target,
        eigenvectors, eigenvalues, indices_to_track=vec_list,
        fig_name=f"{label}_train_{DATA_OPTION}_f{TARGET_OPTION}",
        lr=LEARNING_RATE, steps=TRAIN_STEPS
    )

    analyze_decay_rate_vs_eigenvalue(
        hist, steps, eigenvalues, vec_list,
        fig_name=f"{label}_decay_rate_{DATA_OPTION}_f{TARGET_OPTION}.png"
    )

    with torch.no_grad():
        y_pred_final = trained_model(x_data)

    function_spectrum(
        y_pred_final, x_data, theta,
        title=f"Final Trained MLP Spectrum ({label})",
        fig_name=f"{label}_final_mlp_spectrum_{DATA_OPTION}_f{TARGET_OPTION}.png"
    )


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
                        help="Data generation option: 'uniform', 'sphered', or 'random'")
    parser.add_argument('--target_func', type=int,
                        default=1, choices=[1, 2, 3])
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--steps', type=int, default=10000,
                        help="Training steps")
    args = parser.parse_args()

    # --- Hyperparameters ---
    INPUT_DIM = args.d_input
    NUM_SAMPLES = args.n_samples
    WIDTH = args.width
    DATA_OPTION = args.data_option
    TARGET_OPTION = args.target_func
    ACTIVATION = args.activation
    LEARNING_RATE = args.lr
    TRAIN_STEPS = args.steps
    LOG_INTERVAL = 100
    file_name = f'_n{NUM_SAMPLES}_w{WIDTH}_{DATA_OPTION}_f{TARGET_OPTION}_{ACTIVATION}_steps{TRAIN_STEPS}_lr{LEARNING_RATE}'

    SAVE_DIR = create_save_dir('experiments/pipeline2', suffix=file_name)
    print('Current run path: ', SAVE_DIR)

    torch.manual_seed(42)
    np.random.seed(42)
    vec_list = list(range(0, 10)) + [i*10 for i in range(1, 6)]

    # --- generate data and target function ---
    x_data, theta = init_inputs(num_inputs=NUM_SAMPLES,
                                dim=INPUT_DIM, option=DATA_OPTION)
    y_target = get_target_function(x_data, theta, TARGET_OPTION).unsqueeze(1)

    function_spectrum(
        y_target, x_data, theta, title=f"Target Function (Type {TARGET_OPTION}) Spectrum",
        fig_name=f"target_spectrum_f{TARGET_OPTION}_{DATA_OPTION}.png"
    )

    # --- Empirical NTK analysis ---
    print("--- Empirical NTK analysis ---")
    mlp_for_ntk = MLP_NTK(input_dim=INPUT_DIM,
                          hidden_dim=WIDTH, activation=ACTIVATION)
    empirical_kernel = NTK_empirical(mlp_for_ntk, x_data)
    full_pipeline(empirical_kernel, vec_list, label='EMP')

    # --- Theoretical NTK analysis (only for ReLU + uniform data) ---
    if ACTIVATION == 'ReLU' and DATA_OPTION == 'uniform':
        print("\n--- Theoretical NTK analysis ---")
        analytical_kernel = NTK_analytical_relu(approx_sample=NUM_SAMPLES)
        full_pipeline(analytical_kernel, vec_list, label='ANA')
