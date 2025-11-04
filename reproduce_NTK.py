# Code to reproduce results in *Towards Understanding the Spectral
# Bias of Deep Learning*, Cao et al., 2020. http://arxiv.org/abs/1912.01198
# Mainly reproduce experiments in Section 5.

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gegenbauer
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_act_func


def generate_spherical_data(n_samples, d_input):
    """Generates n_samples points uniformly on the (d_input-1)-sphere."""
    x = torch.randn(n_samples, d_input)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True)
    return x


def get_gegenbauer_poly(degree, dim):
    """
    Returns the appropriate Gegenbauer polynomial function for a given degree and dimension.
    For the sphere S^(d-1), the parameter lambda for Gegenbauer C_k^(lambda) is (d-2)/2.
    """
    # In the paper's notation d=10, so the sphere is S^9.
    # The dimension of the ambient space is 10.
    # Lambda for Gegenbauer polynomials C_n^(lambda) is (dim-2)/2.
    return gegenbauer(degree, (dim - 2) / 2)


class MLP(nn.Module):
    """A simple two-layer MLP with ReLU activation."""

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation='relu'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = get_act_func(activation)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        from IPython import embed
        embed()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def run_experiment(coeffs, title, fig_name='reproduce_1.png'):
    """
    Runs a full training experiment for a given set of coefficients and plots the results.
    """
    print(f"--- Running Experiment: {title} ---")

    # Generate data and fixed vectors (zetas)
    x_train = generate_spherical_data(N_SAMPLES, D_INPUT)

    zetas = []
    for _ in DEGREES:
        zeta = torch.randn(D_INPUT)
        zeta = zeta / torch.linalg.norm(zeta)
        zetas.append(zeta)

    # create the raw (unnormalized) component vectors for each frequency
    raw_components = []
    for i, k in enumerate(DEGREES):
        gegen_poly = get_gegenbauer_poly(k, D_INPUT)
        v_k_raw = gegen_poly(x_train @ zetas[i])
        raw_components.append(v_k_raw)

    # normalize projection vectors
    proj_vectors = [(v / torch.linalg.norm(v)) for v in raw_components]

    # Construct target function y_train by summing the *normalized* components
    y_train = torch.zeros(N_SAMPLES, 1)
    for i in range(len(DEGREES)):
        # y_train += coeffs[i] * proj_vectors[i].unsqueeze(1)
        y_train += coeffs[i] * raw_components[i].unsqueeze(1)

    # Initialize model, loss, and optimizer
    model = MLP(D_INPUT, WIDTH)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Store results
    projection_history = {k: [] for k in DEGREES}
    step_history = []

    # Training loop
    for step in tqdm(range(STEPS)):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if step % LOGGING_INTERVAL == 0:
            with torch.no_grad():
                residual_vec = (y_train - y_pred).squeeze()
                for i, k in enumerate(DEGREES):
                    # Project residual onto the k-th frequency vector
                    # a_k = |<r, v_k>|
                    proj_len = torch.abs(
                        torch.dot(residual_vec, proj_vectors[i]))
                    projection_history[k].append(proj_len.item())
                step_history.append(step)

    # Plotting
    plt.figure(figsize=(8, 6))
    colors = {1: 'blue', 2: 'red', 4: 'green'}
    for k in DEGREES:
        plt.plot(step_history,
                 projection_history[k], label=f'k={k}', color=colors[k])

    plt.xlabel("Step")
    plt.ylabel(r"Projection Length $\hat{a}_k$")
    plt.title(f"Learning Dynamics: {title}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # A log scale on the y-axis helps visualize the different decay rates clearly
    # plt.yscale('log')
    plt.savefig(fig_name)


if __name__ == "__main__":
    # --- paper's setting ---
    D_INPUT = 10       # Input dimension
    N_SAMPLES = 1000   # Number of data points
    WIDTH = 4096        # Width of the MLP
    DEGREES = [1, 2, 4]  # Degrees of spherical harmonics to be used

    # --- My setting ---
    LEARNING_RATE = 1e-3
    STEPS = 20000
    LOGGING_INTERVAL = 100

    # Set a random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Scenario (a): same scale
    coeffs_same_scale = [1.0, 1.0, 1.0]
    title_same_scale = r"$f^*(x) = P_1(x) + P_2(x) + P_4(x)$"
    run_experiment(coeffs_same_scale, title_same_scale, 'reproduce_1a.png')

    # Scenario (b): different scale
    coeffs_diff_scale = [1.0, 3.0, 5.0]
    title_diff_scale = r"$f^*(x) = 1 \cdot P_1(x) + 3 \cdot P_2(x) + 5 \cdot P_4(x)$"
    run_experiment(coeffs_diff_scale, title_diff_scale, 'reproduce_1b.png')
