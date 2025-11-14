'''Implementation of the filtering method proposed by Zhiqin Xu.'''
import numpy as np


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
        kernel_dict (list): A list of 2D NumPy arrays representing the Gaussian kernels to use for filtering.

    Returns:
        tuple: A tuple of two lists, low and high frequency components of the model output
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
    """
    filter_dict = np.linspace(filter_start, filter_end, num=filter_num)
    kernel_dict = normal_kernel(dist, filter_dict)
    f_low, f_high = get_freq_low_high(train_labels, kernel_dict)
    return filter_dict, kernel_dict, f_low, f_high
