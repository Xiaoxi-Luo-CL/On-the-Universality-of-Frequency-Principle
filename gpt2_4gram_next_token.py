# ==========================================================
# Spectral Bias Experiment on NLP Task (4-gram GPT-2)
#
# Combines:
# 1. GPT-2 Model (from transformers)
# 2. N-gram dataset setting (from NgramMLP script)
# 3. Filtering method analysis (from NgramMLP script)
# ==========================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from datasets import load_dataset, DatasetDict
from torch.optim.lr_scheduler import OneCycleLR
import wandb
import math
from tqdm import tqdm
from utils import create_save_dir, set_seed, load_config, plot_diff_distr, calculate_norm, load_spectral_npz
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from mlp_4gram_next_token import load_wikitext, sampled_dataset_to_distance, spectral_diff_training, eval_loader, normal_kernel, get_freq_low_high
from typing import Dict, List, Any, Tuple


def create_ngram_dataset(
    documents: Dataset, tokenizer: AutoTokenizer,
    n: int = 4, ngram_per_doc: int = 30
) -> Dataset:
    """
    Uses datasets.map() to efficiently create the n-gram dataset.
    This replaces the slow, memory-inefficient SampledNgramDataset.__init__
    """
    def process_document_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        tokenized_docs = tokenizer(
            batch['text'], add_special_tokens=False, truncation=False, padding=False)["input_ids"]

        batch_ngrams, batch_targets = [], []

        for ids_list in tokenized_docs:
            ids = np.array(ids_list, dtype=np.int32)
            if len(ids) < n + 1:
                continue
            max_num = len(ids) - n
            k = min(ngram_per_doc, max_num)
            if k == 0:
                continue
            ngram_indices = np.random.choice(max_num, size=k, replace=False)

            ngrams = ids[ngram_indices[:, None] + np.arange(n)]  # [k, n]
            targets = ids[ngram_indices + n]  # [k]
            batch_ngrams.extend(ngrams.tolist())
            batch_targets.extend(targets.tolist())

        return {"input_ids": batch_ngrams, "labels": batch_targets}

    print(f"Start to create {n}-grams...")
    ngram_dataset = documents.map(
        process_document_batch, batched=True, batch_size=1000, remove_columns=documents.column_names, num_proc=2,
        desc="Building n-grams")
    print(f"Created {len(ngram_dataset)} total n-grams.")

    ngram_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    return ngram_dataset


class NgramDataset(Dataset):
    """
    A simple wrapper for the processed HuggingFace Dataset.
    This is needed for the 'select' method used by the filtering analysis.
    """

    def __init__(self, hf_dataset: Dataset):
        self.hf_dataset = hf_dataset

        print("Loading n-grams into torch tensors for filtering...")
        self.ngrams = hf_dataset['input_ids']
        self.targets = hf_dataset['labels']
        print(f"Loaded {len(self.ngrams)} n-grams.")

    def __len__(self) -> int:
        return len(self.ngrams)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ngrams[idx], self.targets[idx]

    def select(self, indices: np.ndarray):
        """
        Return a new dataset that is a subset of the current one.
        """
        new_ds = self.__class__.__new__(self.__class__)
        new_ds.hf_dataset = None  # This is a subset, not the full hf_dataset

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        new_ds.ngrams = self.ngrams[indices_tensor]
        new_ds.targets = self.targets[indices_tensor]

        return new_ds


class NgramGPT2(nn.Module):
    """
    GPT-2 Model for N-gram prediction.
    Fulfills user request:
    - Official pre-trained GPT-2
    - Frozen embeddings
    - Re-initialized transformer blocks and LM head
    """

    def __init__(self, vocab_size: int, n: int = 4, gpt2_model_name: str = 'gpt2'):
        super().__init__()
        self.n = n
        self.config = GPT2Config.from_pretrained(gpt2_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            gpt2_model_name, config=self.config)

        assert self.model.config.vocab_size == vocab_size
        self.model.resize_token_embeddings(vocab_size)

        self.embed = self.model.transformer.wte

        # Freeze the embedding layer, and re-initialize all other layers
        self.embed.weight.requires_grad = False
        self.model.transformer.wpe.apply(
            self._init_weights)  # Re-init position embeddings
        self.model.transformer.ln_f.apply(
            self._init_weights)  # Re-init final layernorm
        self.model.lm_head.apply(self._init_weights)  # Re-init LM head
        for block in self.model.transformer.h:  # Re-init all transformer blocks
            block.apply(self._init_weights)

    def _init_weights(self, module):
        """
        A standard re-initialization function (from GPT-2's own init).
        This is applied to all layers *except* the frozen wte.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0,
                                  std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_static_embeddings(self) -> torch.Tensor:
        """
        Returns the frozen, pre-trained embedding matrix.
        """
        return self.embed.weight.data.detach().clone()

    def forward(self, input_ids):
        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=None,  # No padding, all tokens attend
            use_cache=False
        )
        hidden_state = transformer_outputs.last_hidden_state[:, -1, :]
        logits = self.model.lm_head(hidden_state)  # [B, V]
        return logits


def main():
    global current_run_path  # Make save path global for helpers

    cfg = load_config('gpt2-config.yaml')
    current_run_path = create_save_dir(folder='NLP-runs/GPT2')
    print('Current run path: ', current_run_path)
    set_seed(cfg["training"]["seed"])

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(project='spectral-gpt2', name=f'first-try', resume='allow')

    print("Loading official GPT-2 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # --- Load Dataset ---
    print(f"Loading dataset wiki ...")
    dataset_dict = load_wikitext(
        num_doc=cfg["dataset"]["num_docs"], test_num=2, train_num=6)
    batch_size = cfg["training"]["batch_size"]

    train_hf_dataset = create_ngram_dataset(
        dataset_dict['train'], tokenizer,
        n=cfg['dataset']['ngram'],
        ngram_per_doc=cfg['dataset']['ngrams_per_doc']
    )
    # Wrap in our simple Dataset for .select()
    train_dataset = NgramDataset(train_hf_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = create_ngram_dataset(
        dataset_dict['test'], tokenizer,
        n=cfg['dataset']['ngram'],
        ngram_per_doc=cfg['dataset']['ngrams_per_doc']
    )
    test_dataset = NgramDataset(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- Define Model ---
    model = NgramGPT2(vocab_size=vocab_size,
                      n=cfg["dataset"]["ngram"]).to(device)

    # --- Compute Frequency Information (Filtering Method) ---
    print('Computing frequency components for target labels ...')
    filter_num = cfg['spectral']['filter_num']
    if cfg['spectral']['scale'] == 'linear':
        filter_var = np.linspace(
            cfg['spectral']['filter_start'], cfg['spectral']['filter_end'], filter_num)
    else:
        filter_var = np.logspace(np.log10(cfg['spectral']['filter_start']),
                                 np.log10(cfg['spectral']['filter_end']), filter_num)

    # Get the STATIC, FROZEN embeddings from the model for distance calculation
    embed_matrix = model.get_static_embeddings().cpu()

    dist, sampled_ngram_dataset, sampled_f_label = sampled_dataset_to_distance(
        train_dataset, embed_matrix,  # type:ignore
        sample=cfg['spectral']['sample_size'], device=device)
    sampled_loader = DataLoader(
        sampled_ngram_dataset, batch_size=batch_size, shuffle=False)

    # try to rescale
    median_dist = torch.median(dist[dist > 1e-6])
    print(f"Original median squared distance: {median_dist:.2f}")
    dist = dist / median_dist

    kernel_stack = normal_kernel(dist, filter_var).to(
        device)  # (filter_num, N, N)
    f_low, f_high = get_freq_low_high(
        kernel_stack, sampled_f_label.to(device), vocab_size, device, label=True)

    # --- 5. Define Optimizer and Loss ---
    loss_fn = nn.CrossEntropyLoss()
    # only train non-frozen parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["lr"])
    )
    scheduler = get_scheduler(
        'linear', optimizer=optimizer, num_warmup_steps=cfg['training']['warmup_steps'], num_training_steps=len(train_loader) * cfg['training']['epochs'])

    # --- 6. Training Loop (from NgramMLP script) ---
    model.train()
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        print(f"--- Epoch {epoch+1}/{cfg['training']['epochs']} ---")
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % 200 == 0:
                wandb.log({'step_loss': loss.item(), 'step': global_step})

            # --- Spectral Analysis Hook ---
            if global_step % cfg['spectral']['interval'] == 0:
                y_pred = eval_loader(model, sampled_loader,
                                     loss_fn, device, return_outputs=True)
                f_train_low, f_train_high = get_freq_low_high(
                    kernel_stack, torch.softmax(y_pred, dim=-1), vocab_size, device, label=False)
                low_diff, high_diff = spectral_diff_training(
                    f_low, f_high, f_train_low, f_train_high, global_step, current_run_path)
                wandb.log({'mean_low_freq_error': np.mean(
                    low_diff), 'mean_high_freq_error': np.mean(high_diff), 'step': global_step})

            global_step += 1

        # --- End of Epoch Evaluation ---
        test_loss, test_acc = eval_loader(model, test_loader, loss_fn, device)

        wandb.log({'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc})
        print(
            f"[Epoch {epoch+1}] Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # --- 7. Final Analysis and Save ---
    wandb.finish()
    print('Plotting spectral results ...')
    lowdiff_arr, highdiff_arr = load_spectral_npz(
        f'{current_run_path}/spectral_log')

    for fid in range(filter_num):
        plot_diff_distr(filter_var[fid],  lowdiff_arr[:, fid],
                        highdiff_arr[:, fid], current_run_path)

    torch.save(model.state_dict(), f'{current_run_path}/GPT2_Ngram.pt')
    print(f"Model saved to {current_run_path}!")


if __name__ == "__main__":
    main()
