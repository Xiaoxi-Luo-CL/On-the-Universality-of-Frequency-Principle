# Empirically Testing the Boundaries of the Frequency Principle

## Introduction
Deep neural networks often fit target functions from low to high frequencies during the training process. Two concurrent and independent work, Xu et al. (2019) and Rahaman et al. (2019), observed and proved this phenomenon, naming it as *Frequency Principle* and *Spectral Bias*,  respectively.

This mechanism is rigorously demonstrated for DNNs of general settings (any commonly used activation function, general population density of data, a general class of loss functions) in a subsequent work (Luo et al., 2019). The connection between Frequency Principle of over-parameterized two-layer Relu MLP and NTK was further built. (Cao et al., 2020) 

The concept of Frequency Principle is appealing because it is strongly related to generalization of DNNs:
The lower frequency components of trained networks are more robust to random parameter perturbations (Rahaman et al. 2019), and according to Frequency Principle, among all the functions that can fit the training data, a DNN is implicitly biased during the training towards a function with more power at low frequencies. (Xu et al. 2019)

However, several questions cannot be well answered by existing theories and empirical results:
+ Although Frequency Principle is proved in a generalized setting, but several assumptions are made on the loss function/data distribution/activation function. (Luo et al., 2019) Most importantly, the widely used Cross Entropy Loss does not meet their assumptions.
+ Existing experiments for verifying Frequency Principle are mostly on image classification tasks, and few of them are on text-based tasks, e.g., next-token prediction.
+ It is unclear whether similar experiments could be extended to Transformers. 

Therefore, it is a natural to ask: To what extent does Frequency Principle holds? Is it a property of MLPs (under restricted settings), or could be extended to a wider range of training settings, model architectures, and real-life tasks?

The aim of this project is to **empirically test the boundaries of Frequency Principle**.
Experiments are structured into three progressive stages:
1. Based on Cao et al. (2020), I verify the theoretical connection between NTK and Frequency Principle, and find that under lazy training regime, Frequency Principle could be fully predicted by NTK theory. I also extend the analyses from ideal uniform distribution to non-uniform distributions and even feature learning regime, where the connection between NTK and Frequency Principle is broken, but Frequency Principle still holds.
2. I validate Frequency Principle on 4-gram next-prediction task on MLP, with pre-trained embedding. 
3. I conduct 4-gram next-prediction experiments on GPT-2.

## Code
+ Experiment 1: `experiments.py`, showing the connection between NTK and Frequency Principle by (a) projecting the residual on the eigenvectors of NTK and (b) directly decomposing the MLP.
+ Experiment 2: `mlp_4gram_next_token.py`
+ Experiment 3: `gpt2_4gram_next_token.py`

Other code:
+ `reproduce_NTK.py`: Reproduce results in Cao et al. (2020), Section 5.
+ `fprinciple-1d.py`: Experiments to verify Frequency Principle with 1D data, re-written from https://github.com/xuzhiqin1990/dnn_simple_experiments/tree/main/frequency_principle.
+ `fprinciple-nd.py`: Experiments to verify Frequency Principle with digh-dimensional data (CIFAR image classification), re-written from https://github.com/xuzhiqin1990/dnn_simple_experiments/tree/main/frequency_principle.
+ `decompose_NTK.py`: Eigen-decompose empirical NTK matrix. Reference: https://github.com/genglinliu/NTK-study/tree/main.
+ `check_nufft.py`: compare the original signal and the decomposed results by NUDFT.

## Reference

Yuan Cao, Zhiying Fang, Yue Wu, Ding-Xuan Zhou, and Quanquan Gu. Towards understanding
the spectral bias of deep learning, 2020. URL https://arxiv.org/abs/1912.01198.

Tao Luo, Zheng Ma, Zhi-Qin John Xu, and Yaoyu Zhang. Theory of the frequency principle for
general deep neural networks, 2019. URL https://arxiv.org/abs/1906.09235.

Zhi-Qin John Xu, Yaoyu Zhang, and Yanyang Xiao. Training behavior of deep neural network in
frequency domain, 2019. URL https://arxiv.org/abs/1807.01251.

Zhi-Qin Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier
analysis sheds light on deep neural networks. Communications in Computational Physics, 28(5):
1746–1767, June 2020.
