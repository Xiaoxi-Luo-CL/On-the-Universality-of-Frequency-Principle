'''Calculate NTK in three ways: Infinite analytical, infinite empirical, and finite empirical. Decompose NTK into eigenvalues and eigenvectors.'''

import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.func import vmap, jacrev
from utils import relu, d_relu, sin, cos, d_cos, sigmoid, d_sigmoid, tanh, d_tanh, MLP_General


def init_inputs(num_inputs=200):
    theta = np.linspace(0.0, 2 * np.pi, num=num_inputs)
    x = np.array((np.cos(theta), np.sin(theta)))
    return x.transpose()  # (100, 2)


def check(matrix, tol=1e-8):
    # check kernel symmetry and positive definitiveness
    if not np.all(np.abs(matrix-matrix.T) < tol):
        raise ValueError("warning: kernel is not symmetric")
    if not np.all(np.linalg.eigh(matrix)[0] >= -1e-2):
        raise ValueError("warning: kernel is not positive semi-definite")


# --- Infinite-Width NTK Calculations (analytical and approximated) ---

def NTK_analytical_relu(ignore_warning=False, approx_sample=None):
    """
    analytic solution: u = <x, x'> / ||x|| ||x'||
    k(u) = u k_0(u) + k_1(u)
    k_0 = 1/pi (pi - arccos(u))
    k_1 = 1/pi (u (pi - arccos(u)) + sqrt(1 - u^2))

    Since ||x|| = 1, we have u = <x, x'>
    """
    def kappa(u):
        pi = np.pi
        k_0 = (1/pi) * (pi - np.arccos(u))
        k_1 = (1/pi) * (u * (pi - np.arccos(u)) + np.sqrt(1 - u**2))
        return u * k_0 + k_1
    if approx_sample is None:
        approx_sample = N_SAMPLES
    x = init_inputs(approx_sample)  # (100, 2)
    inner_prod = np.dot(x, x.T)  # (100, 100)
    # numerical fix => values slightly > 1 will be capped
    inner_prod[inner_prod > 1.0] = 1.0
    kernel = kappa(inner_prod)
    if not ignore_warning:
        check(kernel)
    return kernel


def infinite_NTK_approx(activation='relu', ignore_warning=False, num_w=40000):
    """
    Numerical kernel approximation. Reference: 
    https://papers.nips.cc/paper/2019/file/c4ef9c39b300931b69a36fb3dbb8d60e-Paper.pdf
    equation (3) on page 3 and section 3.3 on page 7 
    K(x, x') = <x, x'> E[sig'(<w, x>) sig'(<w, x'>)] + E[sig(<w, x>) sig(<w, x'>)]
    Take expectation over w ~ N(0, 1).

    activation = ['relu', 'sin', 'cos']
    """
    x = init_inputs(num_inputs=N_SAMPLES)  # (100, 2)

    # init kernel and weight
    w1, w2 = np.random.normal(
        0.0, 1.0, size=[2, num_w, D_INPUT])  # (num_w, 2) both

    xx = np.dot(x, x.T)  # (100, 100)
    xx[xx > 1.0] = 1.0
    w1x = np.dot(w1, x.T)  # (num_w, 100)
    w2x = np.dot(w2, x.T)  # (num_w, 100)

    activation_func, d_activation = act_map[activation]

    sigma_w1x = activation_func(w1x)  # (num_samples, 100)
    sigma_w2x = d_activation(w2x)  # (num_samples, 100)

    expection_1 = np.dot(sigma_w1x.T, sigma_w1x) / num_w
    expection_2 = np.dot(sigma_w2x.T, sigma_w2x) / num_w
    kernel = 2 * xx * expection_2 + 2 * expection_1

    if not ignore_warning:
        check(kernel)
    return kernel


# --- Empirical NTK Calculation for *Finite-Width* MLP ---

def NTK_empirical(model, x, ignore_warning=False):
    """
    Computes the empirical NTK for a given finite-width model and data. 
    Reference: https://docs.pytorch.org/tutorials/intermediate/neural_tangent_kernels.html#compute-the-ntk-method-1-jacobian-contraction
    """
    print("Computing Empirical NTK for the finite-width model")
    model.eval()
    if type(x) is np.ndarray:
        x = torch.Tensor(x)

    def fnet(params, x_input):
        return torch.func.functional_call(model, params, x_input.unsqueeze(0))
    compute_jacobian_for_sample = jacrev(fnet, argnums=0)

    # use vmap to parallel the Jacobian calculation on dimension x
    # in_dims=(None, 0) means mapping on the first (batch) dimesion
    jacobians_dict = vmap(compute_jacobian_for_sample, in_dims=(
        None, 0))(dict(model.named_parameters()), x)

    # Concat every parameter's gradient into the Jacobian
    jacobians_flat_list = [j.flatten(start_dim=1)  # (num_inputs, *param_shape)
                           for j in jacobians_dict.values()]
    # (num_inputs, num_params)
    jacobian_matrix = torch.cat(jacobians_flat_list, dim=1)
    # NTK = J @ J.T
    kernel = (jacobian_matrix @ jacobian_matrix.T).detach().numpy()

    if not ignore_warning:
        check(kernel)
    return kernel


def plot_eigendecay(kernel_ls: list, label_ls: list, loglog=False, eigen_num=200, fig_name='eigendecay', fig_path=None, sorted_eigenval: list | None = None):
    for i, (kernel, label) in enumerate(zip(kernel_ls, label_ls)):
        if sorted_eigenval is not None:
            sorted_eigenvalues = sorted_eigenval[i]
        else:
            eigenvalues = np.linalg.eigh(kernel)[0]
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        if loglog:
            plt.loglog(sorted_eigenvalues[:eigen_num], label=label)
        else:
            plt.yscale('log')
            plt.plot(sorted_eigenvalues[:eigen_num], label=label)
    plt.legend()
    plt.title(fig_name)
    fig_path = fig_name if fig_path is None else fig_path
    plt.savefig(fig_path+'.png')
    plt.clf()


def NTK_empirical_mlp(act_func='ReLU'):
    model = MLP_General(input_dim=D_INPUT,
                        hidden_dim=WIDTH, activation=act_func)
    x = init_inputs(num_inputs=N_SAMPLES)  # (2, 100)
    kernel = NTK_empirical(model, x)
    return kernel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_input', type=int, default=2,
                        help='Input dimension')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of data points')
    parser.add_argument('--width', type=int, default=2000,
                        help='Width of the MLP')
    args = parser.parse_args()
    D_INPUT = args.d_input       # Input dimension
    N_SAMPLES = args.n_samples   # Number of data points
    WIDTH = args.width
    # ------
    act_map = {'relu': (relu, d_relu), 'sin': (sin, cos), 'cos': (cos, d_cos),
               'sigmoid': (sigmoid, d_sigmoid), 'tanh': (tanh, d_tanh)}

    kernel_analytical = NTK_analytical_relu()
    kernel_relu = infinite_NTK_approx('relu')
    kernel_sin = infinite_NTK_approx('sin')
    kernel_cos = infinite_NTK_approx('cos')
    kernel_sigmoid = infinite_NTK_approx('sigmoid')
    kernel_tanh = infinite_NTK_approx('tanh')
    # empirical check with MLP NTK
    emperical_relu = NTK_empirical_mlp(act_func='ReLU')

    affix = f'w{WIDTH}_n{N_SAMPLES}'
    plot_eigendecay(kernel_ls=[kernel_analytical, kernel_relu, kernel_sin,
                               kernel_cos, kernel_sigmoid, kernel_tanh],
                    label_ls=['relu_analytical', 'relu',
                              'sin', 'cos', 'sigmoid', 'tanh'],
                    fig_name=f'eigendecay_{affix}')

    plot_eigendecay(kernel_ls=[kernel_analytical, kernel_relu, emperical_relu],
                    label_ls=['analytical_relu', 'relu', 'empirical_relu'],
                    fig_name=f'relu_{affix}')
