import numpy as np
import torch
import matplotlib.pyplot as plt
import finufft
import math
from typing import Dict, List, Tuple, Any
import torch.nn as nn
import os


def plot_spectral_error_heatmap(
    relative_error_history: Dict[int, List[float]],
    step_history: list, save_dir: str,
    fig_name: str = "spectral_error_dynamics_heatmap.png"
):
    """
    Plots the relative error dynamics as a heatmap, where each frequency
    magnitude |k| is a horizontal band.

    Inspired by the 'plot_diff_distr' function provided by the user.
    """
    # ---  Data Preparation ---
    # Sort the frequency magnitudes to ensure low-freqs are at the bottom
    tracking_mags = sorted(relative_error_history.keys())

    # Create the 2D data matrix: (num_bands, num_steps)
    error_matrix = np.array([relative_error_history[k]
                             for k in tracking_mags])
    num_bands, num_steps = error_matrix.shape

    # Check if history and steps match
    assert num_steps == len(
        step_history), f'Mismatching in step history ({len(step_history)}) and error history ({num_steps})'

    # --- Plotting ---
    plt.figure(figsize=(10, num_bands))
    ax = plt.gca()

    # We use imshow for a clean grid plot.
    # origin='lower' places the first band (lowest |k|) at the bottom.
    im = ax.imshow(
        error_matrix,
        aspect='auto',
        origin='lower',
        # Set extent to match the training steps
        extent=[step_history[0], step_history[-1], -0.5, num_bands - 0.5],
        vmin=0.0,
        vmax=2.0  # Relative error is normalized between 0 and (approx) 1
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Frequency Magnitude $|k|$")
    ax.set_yticks(np.arange(num_bands))
    ax.set_yticklabels([f"$|k|={mag}$" for mag in tracking_mags])
    ax.set_xticks(step_history)

    plt.colorbar(im, ax=ax)
    ax.set_title(f"Relative Error in Frequency Domain")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.clf()
    plt.close()


def _calculate_nufft_spectrum(y_values: np.ndarray, x_data: torch.Tensor,
                              theta: torch.Tensor | None, option: str) -> dict:
    """
    Calculate the spectrum (NUFFT or FFT) and return frequencies.
    (Version 3: Corrected to return normalized AMPLITUDE, not POWER)
    """
    N_samples = len(y_values)
    frequency_map = {}  # {k: amplitude}

    if option == 'uniform':
        fft_vals = np.fft.fft(y_values)
        amp_raw = fft_vals / N_samples

        # We only care about positive frequencies k=0, 1, ..., N/2
        number = min(N_samples // 2, 50)
        amplitudes = amp_raw[:number]
        freqs = np.arange(number)

        for k, amp in zip(freqs, amplitudes):
            frequency_map[k] = amp
        return frequency_map

    elif option in ['sphered', 'random']:
        n_freqs = 21  # Target frequency grid size [-10, 10]

        if option == 'sphered':
            # 1D NUFFT on angle theta
            theta_np = theta.numpy() - np.pi  # Rescale to [-pi, pi]
            f_k = finufft.nufft1d1(theta_np, y_values.astype(
                np.complex64), n_freqs, isign=-1)

            freqs_shifted = np.arange(n_freqs) - n_freqs // 2

            # Calculate normalized AMPLITUDE |F(k)| / N
            amp_shifted = f_k / N_samples

            for k, amp in zip(freqs_shifted, amp_shifted):
                frequency_map[int(k)] = amp
            return frequency_map

        else:
            x_scaled = np.ascontiguousarray(x_data[:, 0].numpy())
            y_scaled = np.ascontiguousarray(x_data[:, 1].numpy())

            f_k = finufft.nufft2d1(
                x_scaled, y_scaled, y_values.astype(np.complex64),
                (n_freqs, n_freqs), isign=-1)

            # Calculate normalized AMPLITUDE |F(k)| / N
            amp_2d = f_k / N_samples
            kx = np.arange(n_freqs) - n_freqs // 2
            ky = np.arange(n_freqs) - n_freqs // 2

            for i, kx_val in enumerate(kx):
                for j, ky_val in enumerate(ky):
                    mag_k = int(kx_val**2 + ky_val**2)
                    if mag_k not in frequency_map:
                        frequency_map[mag_k] = []
                    frequency_map[mag_k].append(np.abs(amp_2d[j, i]))

            # Average amplitude over each magnitude |k|
            final_map = {mag_k: np.mean(amps)
                         for mag_k, amps in frequency_map.items()}
            return final_map
    else:
        raise ValueError("Unsupported data option for spectral analysis.")


def analyze_spectral_error_dynamics(
    x_data: torch.Tensor, y_target: torch.Tensor,
    y_pred_history: List[np.ndarray], theta: torch.Tensor | None,
    steps: list, data_option: str, save_dir: str
):
    """Analyzes relative error in frequency domain during training"""
    # target spectrum (ground truth)
    target_values = y_target.squeeze().detach().numpy()
    target_freq_map = _calculate_nufft_spectrum(
        target_values, x_data, theta, data_option)

    # track relative error, sorted by magnitude |k|, skip 0
    k_magnitudes = sorted(target_freq_map.keys())
    tracking_mags = [k for k in k_magnitudes if np.abs(
        target_freq_map[k]) > 1e-4]
    if len(tracking_mags) > 10:
        tracking_mags = [k for k in tracking_mags if k >= 0][:8]
    from IPython import embed
    embed()

    # store relative error: {mag_k: [error_step0, error_step1, ...]}
    relative_error_history: Dict[int, List[float]] = {
        mag: [] for mag in tracking_mags}
    target_power_map = {mag: target_freq_map[mag] for mag in tracking_mags}

    for y_pred_np in y_pred_history:
        y_pred_np_squeezed = y_pred_np.squeeze()
        # calculate spectrum of y_pred
        pred_freq_map = _calculate_nufft_spectrum(
            y_pred_np_squeezed, x_data, theta, data_option)

        # calculate relative error for tracked magnitudes
        for mag_k in tracking_mags:
            P_target = target_power_map[mag_k]
            # Use 0 if frequency not found in predicted spectrum (rare)
            P_pred = pred_freq_map.get(mag_k, 0.0)

            # Relative Error: |P_target - P_pred| / P_target
            relative_error = np.abs(P_target - P_pred) / \
                (np.abs(P_target) + 1e-10)
            relative_error_history[mag_k].append(relative_error)

    # 4. Plotting
    plt.figure(figsize=(10, 7))
    for mag_k in tracking_mags:
        plt.plot(steps, relative_error_history[mag_k],
                 label=f"$|k|={mag_k}$")

    plt.xlabel("Training Step")
    plt.ylabel("Relative Error ($|P_{target} - P_{pred}| / P_{target}$)")
    plt.title(f"Spectral Bias: Relative Error ({data_option})")
    plt.yscale('log')
    plt.legend(title="Frequency Magnitude $|k|$")
    plt.grid(True, which="both", linestyle='--')
    plt.savefig(f'{save_dir}/spectral_error_dynamics.png')
    plt.clf()

    # plot_spectral_error_heatmap(relative_error_history, steps, save_dir)
