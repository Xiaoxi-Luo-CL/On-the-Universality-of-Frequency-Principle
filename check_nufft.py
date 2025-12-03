#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone script to validate the accuracy of 2D Non-Uniform FFT (FINUFFT).
"""
import numpy as np
import matplotlib.pyplot as plt
import finufft
import argparse
import sys


def generate_nonuniform_data(N_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates N_samples non-uniform 2D points in the range [-pi, pi].
    FINUFFT requires input coordinates to be within this periodic box.
    """
    print(f"Generating {N_samples} non-uniform 2D data points...")
    # Generate points uniformly in the [-pi, pi] box
    x_data = np.random.uniform(-np.pi, np.pi, N_samples)
    y_data = np.random.uniform(-np.pi, np.pi, N_samples)

    # Ensure C-contiguous memory layout for finufft
    x_data = np.ascontiguousarray(x_data, dtype=np.float64)
    y_data = np.ascontiguousarray(y_data, dtype=np.float64)

    return x_data, y_data


def run_nufft(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    n_freqs: int
) -> np.ndarray:
    """
    Runs the 2D NUFFT Type 1 (non-uniform to uniform).
    """
    print("Running finufft.nufft2d1...")
    if n_freqs % 2 == 0:
        print(
            f"Warning: n_freqs={n_freqs} is even. Using n_freqs+1 for a centered grid.")
        n_freqs += 1

    try:
        # isign=-1 corresponds to the standard e^{-ikx} Fourier transform
        spectrum = finufft.nufft2d1(
            x_coords, y_coords, values.astype(np.complex128),
            (n_freqs, n_freqs), isign=-1
        )
    except Exception as e:
        print(f"FINUFFT Error: {e}")
        print("Please ensure finufft is installed correctly.")
        sys.exit(1)

    return spectrum


def run_nufft(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    n_freqs: int
) -> np.ndarray:
    """
    Runs the 2D NUFFT Type 1 (non-uniform to uniform).

    Args:
        x_coords (N,): Non-uniform x coordinates in [-pi, pi].
        y_coords (N,): Non-uniform y coordinates in [-pi, pi].
        values (N,): Signal values at (x, y). Must be complex.
        n_freqs (int): Number of frequency modes (K) to compute.
                       Grid will be (K, K). Must be odd for easy centering.

    Returns:
        np.ndarray: (K, K) complex array of frequency coefficients.
    """
    print("Running finufft.nuffufft2d1...")
    if n_freqs % 2 == 0:
        print(
            f"Warning: n_freqs={n_freqs} is even. Using n_freqs+1 for a centered grid.")
        n_freqs += 1

    try:
        # isign=-1 corresponds to the standard e^{-ikx} Fourier transform
        spectrum = finufft.nufft2d1(
            x_coords, y_coords, values.astype(np.complex128),
            (n_freqs, n_freqs), isign=-1
        )
    except Exception as e:
        print(f"FINUFFT Error: {e}")
        print("Please ensure finufft is installed correctly.")
        sys.exit(1)

    return spectrum


def get_frequency_grid(n_freqs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the centered k-space grid.
    Ensures n_freqs is ODD for a perfect k=0 center.
    Returns: kx_grid, ky_grid, k_vec_1d
    """
    if n_freqs % 2 == 0:
        # Ensure grid size is odd
        n_freqs += 1

    # k-vectors from -n_freqs//2 to n_freqs//2
    k_vec_1d = np.arange(-n_freqs // 2, n_freqs // 2 + 1)
    # Use 'xy' indexing (default) for meshgrid
    kx_grid, ky_grid = np.meshgrid(k_vec_1d, k_vec_1d)
    return kx_grid, ky_grid, k_vec_1d


def find_nearest_k_index(k_vec_1d: np.ndarray, k_val: int) -> int:
    """
    Helper to find the grid index for a k-value.
    Assumes k_vec_1d is sorted: [-K, -K+1, ..., 0, ..., K-1, K].
    """
    if k_val < k_vec_1d[0] or k_val > k_vec_1d[-1]:
        print(
            f"Warning: k_val={k_val} is outside the frequency grid range [{k_vec_1d[0]}, {k_vec_1d[-1]}]. Clamping.")
    index = np.argmin(np.abs(k_vec_1d - k_val))
    return index


def plot_spectrum_comparison(
    fig,
    row: int,
    empirical_spectrum: np.ndarray,
    theoretical_spectrum: np.ndarray,
    title: str,
    n_freqs: int,
    N_samples: int
):
    """
    Plots the empirical (NUFFT) and theoretical spectra side-by-side.
    (Version 4: Corrected normalization and amplitudes)
    """
    print(f"Plotting comparison for: {title}")

    if n_freqs % 2 == 0:
        n_freqs += 1

    # --- 1. Empirical (NUFFT) Plot (FIXED) ---
    ax1 = fig.add_subplot(3, 2, row*2 + 1)

    # Normalize by N_samples to estimate the Fourier coefficients
    emp_amplitude = np.abs(empirical_spectrum) / N_samples

    k_min = -n_freqs // 2
    k_max = n_freqs // 2

    # Set vmax based on theoretical plot for consistent scale
    vmax = np.max(np.abs(theoretical_spectrum)) * 1.2
    if vmax < 1e-9:  # Handle zero theoretical spectrum
        vmax = np.max(emp_amplitude) * 1.2
    if vmax < 1e-9:  # Handle all zero
        vmax = 1.0

    im1 = ax1.imshow(
        emp_amplitude,
        origin='lower',
        extent=(k_min, k_max, k_min, k_max),
        aspect='auto',
        vmin=0, vmax=vmax
    )
    fig.colorbar(im1, ax=ax1)
    ax1.set_title(f"{title} (Empirical NUFFT)")
    ax1.set_xlabel("$k_x$")
    ax1.set_ylabel("$k_y$")

    # --- 2. Theoretical Plot (FIXED) ---
    ax2 = fig.add_subplot(3, 2, row*2 + 2)

    theo_amplitude = np.abs(theoretical_spectrum)

    im2 = ax2.imshow(
        theo_amplitude,
        origin='lower',
        extent=(k_min, k_max, k_min, k_max),
        aspect='auto',
        vmin=0, vmax=vmax
    )
    fig.colorbar(im2, ax=ax2)
    ax2.set_title(f"{title} (Theoretical)")
    ax2.set_xlabel("$k_x$")
    ax2.set_ylabel("$k_y$")


def main(args):
    """
    Main function to run the NUFFT validation tests.
    """
    x_coords, y_coords = generate_nonuniform_data(args.n_samples)

    # Use n_freqs from args, but get_frequency_grid will ensure it's odd
    n_freqs_input = args.n_freqs
    kx_grid, ky_grid, k_vec_1d = get_frequency_grid(n_freqs_input)

    # Use the actual (potentially incremented) grid size
    n_freqs_actual = len(k_vec_1d)

    fig = plt.figure(figsize=(12, 18))
    N_samples = args.n_samples  # Get N for normalization

    # --- Test 1: Low-Frequency (Shifted Gaussian) ---
    title_1 = "Test 1: Low-Freq Gaussian $e^{-0.5(x^2+y^2)}$"
    values_1 = np.exp(-0.5 * (x_coords**2 + y_coords**2))
    emp_spec_1 = run_nufft(x_coords, y_coords, values_1, n_freqs_actual)

    # Analytical solution (relative amplitude)
    theo_spec_1 = np.exp(-0.5 * (kx_grid**2 + ky_grid**2))

    plot_spectrum_comparison(
        fig, 0, emp_spec_1, theo_spec_1, title_1, n_freqs_actual, N_samples)

    # --- Test 2: High-Frequency (Shifted Gaussian) ---
    title_2 = "Test 2: High-Freq Gaussian $e^{-0.5(x^2+y^2)} \cdot \cos(8x)$"
    values_2 = np.exp(-0.5 * (x_coords**2 + y_coords**2)) * \
        np.cos(8 * x_coords)
    emp_spec_2 = run_nufft(x_coords, y_coords, values_2, n_freqs_actual)

    # cos(8x) = 0.5 * (e^i8x + e^-i8x)
    # The amplitudes of the two shifted Gaussians are 0.5
    gauss_transform = 0.5 * np.exp(-0.5 * (kx_grid**2 + ky_grid**2))
    k_vec = k_vec_1d
    k_shift = find_nearest_k_index(k_vec, 8) - find_nearest_k_index(k_vec, 0)

    theo_spec_2 = (
        np.roll(gauss_transform, k_shift, axis=1) +
        np.roll(gauss_transform, -k_shift, axis=1)
    )
    plot_spectrum_comparison(
        fig, 1, emp_spec_2, theo_spec_2, title_2, n_freqs_actual, N_samples)

    # --- Test 3: Dirac Deltas (User's Suggestion) ---
    title_3 = "Test 3: $ \cos(2x) + \cos(4y) + 5\sin(3x) $"
    values_3 = np.cos(2 * x_coords) + np.cos(4 * y_coords) + \
        5 * np.sin(3 * x_coords)
    emp_spec_3 = run_nufft(x_coords, y_coords, values_3, n_freqs_actual)

    # Create theoretical grid of deltas
    theo_spec_3 = np.zeros(
        (n_freqs_actual, n_freqs_actual), dtype=np.complex128)
    k_vec = k_vec_1d

    idx_p2_x = find_nearest_k_index(k_vec, 2)
    idx_n2_x = find_nearest_k_index(k_vec, -2)
    idx_p3_x = find_nearest_k_index(k_vec, 3)
    idx_n3_x = find_nearest_k_index(k_vec, -3)
    idx_p4_y = find_nearest_k_index(k_vec, 4)
    idx_n4_y = find_nearest_k_index(k_vec, -4)
    idx_z_x = find_nearest_k_index(k_vec, 0)
    idx_z_y = find_nearest_k_index(k_vec, 0)

    # Use correct Fourier Series amplitudes
    # cos(2x) -> 0.5 at kx = +/- 2
    theo_spec_3[idx_z_y, idx_p2_x] = 0.5
    theo_spec_3[idx_z_y, idx_n2_x] = 0.5
    # cos(4y) -> 0.5 at ky = +/- 4
    theo_spec_3[idx_p4_y, idx_z_x] = 0.5
    theo_spec_3[idx_n4_y, idx_z_x] = 0.5
    # 5*sin(3x) -> -2.5j at kx=3, +2.5j at kx=-3
    # Amplitude |coeff| is 2.5
    theo_spec_3[idx_z_y, idx_p3_x] = -2.5j
    theo_spec_3[idx_z_y, idx_n3_x] = 2.5j

    plot_spectrum_comparison(
        fig, 2, emp_spec_3, theo_spec_3, title_3, n_freqs_actual, N_samples)

    # --- Save Figure ---
    plt.tight_layout()
    save_name = f"nufft_validation_N{args.n_samples}_K{n_freqs_actual}.png"
    plt.savefig(save_name)
    print(f"\nSaved validation plot to: {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D NUFFT Sanity Check Script"
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=5000,
        help="Number of non-uniform samples (N). (Default: 5000)"
    )
    parser.add_argument(
        '--n_freqs',
        type=int,
        default=41,
        help="Frequency grid size (K). KxK grid. (Default: 41, gives [-20, 20] range)"
    )

    # Per your request, comments are in English
    args = parser.parse_args()

    main(args)
