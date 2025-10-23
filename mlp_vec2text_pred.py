# ==========================================================
# Spectral Bias Experiment on NLP Task (4-gram MLP with Filtering Method)
# Implements:
# * Word2Vec embeddings (pre-trained)
# * 4-gram next-token prediction
# * MLP configurable via YAML
# * Filtering method analysis (frequency decomposition via Gaussian kernel)
# ==========================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import gensim.downloader as api
import re
import argparse
import yaml
import random
from typing import Union
import os
import json
from utils import get_act_func, create_save_dir, set_seed


def load_config():
    parser = argparse.ArgumentParser(
        description="Spectral Bias NLP Experiment")
    parser.add_argument(
        "--config", type=str, default="nlp-config.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int,
                        help="Override number of training epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--sample_docs", type=int,
                        help="Override number of sampled documents")
    parser.add_argument("--ngrams_per_doc", type=int,
                        help="Override n-grams per document")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.sample_docs is not None:
        cfg["dataset"]["num_docs"] = args.sample_docs
    if args.ngrams_per_doc is not None:
        cfg["dataset"]["ngrams_per_doc"] = args.ngrams_per_doc

    return cfg


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


def tokenize(text: str):
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return text.lower().split()


# Model definition ------------------------------------------

class NgramMLP(nn.Module):
    def __init__(self, embedding_matrix, hidden_dims, output_dim, activation="relu", n=4):
        super().__init__()
        act_fn = get_act_func(activation.lower())
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True
        )
        input_dim = (n - 1) * embedding_matrix.shape[1]

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), act_fn]
            last_dim = h
        layers += [nn.Linear(last_dim, output_dim), nn.Softmax(dim=-1)]
        self.net = nn.Sequential(*layers)

    def forward(self, input_ids):
        embeds = self.embed(input_ids)  # [B, n-1, dim]
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
        from IPython import embed
        embed()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# Filtering Method Components -------------------------

def compute_distance_matrix(embeddings, metric="euclidean"):
    """Compute pairwise distance matrix for input embeddings. Default metric is Euclidean."""
    X = embeddings
    if metric == "euclidean":
        dist = -2 * np.dot(X, X.T) + np.sum(X**2,
                                            axis=1)[:, None] + np.sum(X**2, axis=1)
        return dist
    elif metric == "cosine":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / (norm + 1e-8)
        sim = np.dot(Xn, Xn.T)
        return 1 - sim
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def normal_kernel(dist, filter_dict):
    """Construct normalized Gaussian kernels for each filter value."""
    kernel_dict = []
    for f in filter_dict:
        kernel = np.exp(-dist / (2 * f))
        mean = np.sum(kernel, axis=1, keepdims=True)
        kernel_dict.append(kernel / mean)
    return kernel_dict


def gaussian_filter(f_orig, kernel):
    """Apply Gaussian filter to given signal or embedding set."""
    return np.matmul(kernel, f_orig)


def get_freq_low_high(yy, kernel_dict):
    """Decompose signals into low and high frequency components."""
    f_low, f_high = [], []
    for kernel in kernel_dict:
        f_new = gaussian_filter(yy, kernel)
        f_low.append(f_new)
        f_high.append(yy - f_new)
    return f_low, f_high


# Main -----------------------------------------------------

def main():
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
        dataset['train'], word2id, n=cfg['dataset']['ngram'])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    test_dataset = SampledNgramDataset(
        dataset['test'], word2id, n=cfg['dataset']['ngram'])
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 4. Define model
    model = NgramMLP(
        embedding_matrix=embedding_matrix,
        hidden_dims=cfg["model"]["hidden_dims"],
        output_dim=len(vocab),
        activation=cfg["model"]["activation"],
        n=cfg["dataset"]["ngram"]
    ).to(device)

    loss_fn = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["training"]["lr"])

    # 5. Training loop
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device)
        test_loss = eval_epoch(model, test_loader, loss_fn, device)
        print(
            f"[Epoch {epoch+1}/{cfg['training']['epochs']}] Train Loss={train_loss:.5f}, Test Loss={test_loss:.5f}")

    # 6. Save model & config
    os.makedirs(os.path.dirname(cfg["output"]["save_path"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["output"]["save_path"])
    print(f"Model saved to {cfg['output']['save_path']}!")


if __name__ == "__main__":
    main()
