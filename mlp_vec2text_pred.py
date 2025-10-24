# ==========================================================
# Spectral Bias Experiment on NLP Task (4-gram MLP with Filtering Method)
# Implements:
# * Word2Vec embeddings (pre-trained)
# * 4-gram next-token prediction
# * MLP configurable via YAML
# * Filtering method analysis (frequency decomposition via Gaussian kernel)
# ==========================================================

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import gensim.downloader as api
import re
import wandb
from typing import Union
import os
import json
from utils import get_act_func, create_save_dir, set_seed, load_config, plot_diff_distr


# Load WikiText, build n-gram dataset and dataloader -----------------

def load_wikitext(num_doc=None) -> Union[DatasetDict, dict]:
    dataset = load_dataset("parquet", data_files={
        'test': [f'dataset/wikitext/test-0000{i}-of-00002.parquet' for i in range(1)],
        'train': [f'dataset/wikitext/train-0000{i}-of-00006.parquet' for i in range(3)]},
        streaming=False).select_columns(['text'])
    assert isinstance(dataset, DatasetDict)

    if num_doc is None:
        return dataset
    train_doc_num = min(num_doc, len(dataset['train']))
    test_doc_num = min(num_doc // 4, len(dataset['test']))

    train_indices = np.random.choice(
        len(dataset['train']), size=train_doc_num, replace=False)
    test_indices = np.random.choice(
        len(dataset['test']), size=test_doc_num, replace=False)
    train_dataset_sampled = dataset['train'].select(train_indices)
    test_dataset_sampled = dataset['test'].select(test_indices)

    return {'train': train_dataset_sampled, 'test': test_dataset_sampled}


class SampledNgramDataset(Dataset):
    def __init__(self, documents, word2id, n=4, ngram_per_doc=10, remove_unk=False):
        self.word2id = word2id
        self.n = n
        ngrams_ls, targets_ls = [], []

        for doc in documents:
            tokens = tokenize(doc['text'])
            if len(tokens) < n:
                continue
            ids = np.array([word2id.get(t, word2id['unk']) for t in tokens])
            max_num = len(ids) - n
            ngram_indices = np.random.choice(
                max_num, size=min(ngram_per_doc, max_num), replace=False)
            ngrams = ids[ngram_indices[:, None] + np.arange(n)]
            targets = ids[ngram_indices + n]
            if remove_unk:
                mask = np.all(ngrams != word2id['unk'], axis=1) & (
                    targets != word2id['unk'])
                ngrams = ngrams[mask]
                targets = targets[mask]
            ngrams_ls.append(ngrams)
            targets_ls.append(targets)

        self.ngrams = np.vstack(ngrams_ls)
        self.targets = np.hstack(targets_ls)
        assert self.ngrams.shape[0] == self.targets.shape[0]

        print(f"Constructed {self.ngrams.shape[0]} valid {n}-grams.")

    def __len__(self):
        return self.ngrams.shape[0]

    def __getitem__(self, idx):
        return self.ngrams[idx], self.targets[idx]

    def select(self, indices: np.ndarray):
        """Return a new dataset that is a subset of the current one."""
        new_ds = SampledNgramDataset.__new__(SampledNgramDataset)

        new_ds.word2id = self.word2id
        new_ds.n = self.n

        new_ds.ngrams = self.ngrams[indices]
        new_ds.targets = self.targets[indices]
        return new_ds


def tokenize(text: str):
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return text.lower().split()


# Model definition ------------------------------------------

class NgramMLP(nn.Module):
    def __init__(self, embedding_matrix, hidden_dims, output_dim, activation="relu", n=4):
        super().__init__()
        act_fn = get_act_func(activation.lower())
        self.embed = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone(), freeze=True
        )
        input_dim = n * embedding_matrix.shape[1]

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), act_fn]
            last_dim = h
        layers += [nn.Linear(last_dim, output_dim)]
        #    nn.Softmax(dim=-1)
        self.net = nn.Sequential(*layers)

    def forward(self, input_ids):
        embeds = self.embed(input_ids)  # [B, n, dim]
        x = embeds.view(embeds.size(0), -1)
        return self.net(x)


# Training loop ---------------------------------------------

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(torch.log(y_hat), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_loader(model, loader, loss_fn, device, return_outputs=False):
    '''Evaluate model on given dataloader.'''
    model.eval()
    if return_outputs:
        all_outputs = []
    else:
        total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            if return_outputs:
                all_outputs.append(y_hat.detach().cpu())
            else:
                loss = loss_fn(y_hat, y)
                total_loss += loss.item() * x.size(0)
    if return_outputs:
        return torch.cat(all_outputs, dim=0)
    return total_loss / len(loader.dataset)


# Filtering Method Components -------------------------
def compute_distance_matrix(embeddings: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """
    Compute pairwise distance matrix for embeddings.
    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D)
        metric (str): "euclidean" or "cosine"
    Returns:
        torch.Tensor: Pairwise distance matrix of shape (N, N)
    """
    X = embeddings
    if metric == "euclidean":
        # dist_ij = ||x_i||^2 + ||x_j||^2 - 2 * x_i @ x_j
        x2 = (X ** 2).sum(dim=1, keepdim=True)   # (N, 1)
        dist = x2 + x2.T - 2 * (X @ X.T)
        return dist.clamp_min(0.0)

    elif metric == "cosine":
        # Normalize rows to unit length
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        sim = Xn @ Xn.T
        return 1 - sim
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def normal_kernel(dist, filter_var: list):
    """Construct normalized Gaussian kernels for each filter value."""
    kernel_ls = []
    for var in filter_var:
        kernel = torch.exp(-dist / (2 * var))
        mean = torch.sum(kernel, dim=1, keepdim=True)
        kernel_ls.append(kernel / mean)
    return kernel_ls


def gaussian_filter_label(kernel: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Compute K @ one_hot(labels) without forming the one-hot explicitly.
    Args:
        kernel: torch.Tensor, shape (N, N)
        labels: torch.LongTensor, shape (N,)
        vocab_size: int, V
    Returns:
        f_low: torch.Tensor, shape (N, V), where
               f_low[i, j] = sum_{p: labels[p]==j} kernel[i, p]
    """
    assert kernel.dim() == 2 and kernel.size(
        0) == kernel.size(1), "kernel must be (N,N)"
    N = kernel.size(0)

    # idx: shape (N, N), idx[a, b] = labels[b]
    # labels.unsqueeze(0) -> shape (1, N); expand -> (N, N)
    idx = labels.unsqueeze(0).expand(N, -1)

    # f_low: (N, V), scatter_add along dim=1: a,b -> f_low[a, idx[a,b]] += kernel[a,b]
    f_low = torch.zeros((N, vocab_size))
    f_low.scatter_add_(1, idx, kernel)
    return f_low


def get_freq_low_high(kernel_list: List[torch.Tensor], y: torch.Tensor, vocab_size: int, label=True):
    """
    For a list of kernels (each (N,N)), compute low/high freq components for labels.
    Returns lists f_low_ls, f_high_ls where each element shape is (N, V).
    """
    f_low_ls, f_high_ls = [], []
    if label:
        y = F.one_hot(y.long(), num_classes=vocab_size).float()  # (N, V)
    for kernel in kernel_list:
        f_low = torch.matmul(kernel, y)  # (N, V)
        f_high = y - f_low
        f_low_ls.append(f_low)
        f_high_ls.append(f_high)

    return f_low_ls, f_high_ls


def sampled_dataset_to_distance(ngram_set: SampledNgramDataset, embedding: torch.Tensor, sample=10000) -> Tuple[torch.Tensor, SampledNgramDataset, torch.Tensor]:
    """For a NgramDataset, sample some of them for computing the distance matrix. Convert labels to one-hot vectors."""
    train_indices = np.random.choice(
        len(ngram_set), size=min(len(ngram_set), sample), replace=False)
    sampled_ngram = ngram_set.select(train_indices)

    sampled_ngram_flat = embedding[sampled_ngram.ngrams].reshape(
        sampled_ngram.ngrams.shape[0], -1)
    dist = compute_distance_matrix(sampled_ngram_flat, metric="euclidean")
    return dist, sampled_ngram, torch.Tensor(sampled_ngram.targets)


# Main -----------------------------------------------------

def main():
    # wandb.init(project='spectral-nlp', name=f'mlp-try', resume='allow')
    cfg = load_config()
    seed = cfg["training"]["seed"]
    set_seed(seed)

    device = torch.device(cfg["training"]["device"]
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load vocab and pretrained embeddings
    print(f"Loading pretrained embedding: {cfg['embedding']['name']}")
    w2v = api.load(cfg['embedding']['name'])
    with open(f'dataset/{cfg["embedding"]["name"]}_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    embedding_matrix = torch.tensor(
        np.array([w2v[w] for w in vocab]), dtype=torch.float32)
    word2id = {w: i for i, w in enumerate(vocab)}

    # 2. Load dataset
    print(f"Loading dataset wiki ...")
    dataset = load_wikitext(num_doc=cfg["dataset"]["num_docs"])

    train_dataset = SampledNgramDataset(
        dataset['train'], word2id, n=cfg['dataset']['ngram'], remove_unk=False)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    test_dataset = SampledNgramDataset(
        dataset['test'], word2id, n=cfg['dataset']['ngram'], remove_unk=False)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 3. Compute frequency information for target labels
    filter_num = cfg['spectral']['filter_num']
    filter_var = np.linspace(
        cfg['spectral']['filter_start'], cfg['spectral']['filter_end'], filter_num)
    dist, sampled_ngram_dataset, sampled_f_label = sampled_dataset_to_distance(
        train_dataset, embedding_matrix, sample=10000)
    sampled_loader = DataLoader(
        sampled_ngram_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    kernel_ls = normal_kernel(dist, filter_var)
    f_low, f_high = get_freq_low_high(
        kernel_ls, sampled_f_label, len(vocab), label=True)

    # 4. Define model
    model = NgramMLP(
        embedding_matrix=embedding_matrix,
        hidden_dims=cfg["model"]["hidden_dims"],
        output_dim=len(vocab),
        activation=cfg["model"]["activation"],
        n=cfg["dataset"]["ngram"]).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg["training"]["lr"]))

    # 5. Training loop
    model.train()
    global_step = 0
    lowdiff, highdiff = [[] * filter_num], [[] * filter_num]

    for epoch in range(cfg["training"]["epochs"]):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)  # default reduction: mean
            loss.backward()
            optimizer.step()

            # loss_ls.append(loss.item())
            if global_step % 100 == 0:
                print(f'Step {global_step}:', loss.item())
                # wandb.log({'step_loss': loss.item()})

            if global_step % cfg['spectral']['interval'] == 0:
                y_pred = eval_loader(model, sampled_loader,
                                     loss_fn, device, return_outputs=True)
                assert isinstance(y_pred, torch.Tensor)
                f_train_low, f_train_high = get_freq_low_high(
                    kernel_ls, torch.softmax(y_pred, dim=-1), len(vocab))
                for i in range(filter_num):
                    lowdiff[i].append(np.linalg.norm(
                        f_train_low[i] - f_low[i])/np.linalg.norm(f_low[i]))
                    highdiff[i].append(np.linalg.norm(
                        f_train_high[i] - f_high[i])/np.linalg.norm(f_high[i]))
            global_step += 1
            epoch_loss += loss.item() * x.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)  # type: ignore
        test_loss = eval_loader(model, test_loader, loss_fn, device)
        print(
            f"[Epoch {epoch+1}/{cfg['training']['epochs']}] Train Loss={train_loss:.5f}, Test Loss={test_loss:.5f}")

    # 6. Save model & config
    for filter_idx, filter in enumerate(filter_var):
        lowdiff_ind, highdiff_ind = lowdiff[filter_idx], highdiff[filter_idx]
        plot_diff_distr(filter, lowdiff_ind, highdiff_ind, current_run_path)
    # torch.save(model.state_dict(), f'{current_run_path}/MLP_weight.pt')
    # print(f"Model saved to {cfg['output']['save_path']}!")
    # wandb.finish()


if __name__ == "__main__":
    current_run_path = create_save_dir(folder='NLP-runs')
    print('Current run path: ', current_run_path)
    main()
