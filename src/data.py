"""Data loading and label corruption.

Builds three things:
  * a training set sampled from SST-2 train, with `label` (what the model sees),
    `true_label` (the gold label) and `is_corrupted` (True where we flipped it)
  * the SST-2 validation set (872 examples, clean) used as our in-domain test set
  * a balanced IMDb test subset used for out-of-domain generalization

The corruption is "symmetric" noise: we pick exactly `noise_rate` of the training
examples at random and flip their label (0 <-> 1). With 2 classes a random wrong
label is the same thing as a flip.

The choice of which examples get corrupted depends only on `data_seed`, so every
method (full / LoRA / DoRA) sees exactly the same noisy examples. That is what
makes the memorization comparison fair.
"""

import os

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk


def build_train_set(n_train: int, noise_rate: float, data_seed: int = 0) -> Dataset:
    raw = load_dataset("nyu-mll/glue", "sst2", split="train")
    rng = np.random.default_rng(data_seed)

    idx = rng.permutation(len(raw))[:n_train]
    sub = raw.select(idx.tolist())

    true = np.array(sub["label"])
    n_flip = int(round(noise_rate * n_train))
    flip_idx = rng.permutation(n_train)[:n_flip]
    is_corrupted = np.zeros(n_train, dtype=bool)
    is_corrupted[flip_idx] = True
    noisy = np.where(is_corrupted, 1 - true, true)

    return Dataset.from_dict(
        {
            "uid": [int(i) for i in idx],  # index into the original SST-2 train split
            "text": sub["sentence"],
            "label": noisy.tolist(),
            "true_label": true.tolist(),
            "is_corrupted": is_corrupted.tolist(),
        }
    )


def build_sst2_test() -> Dataset:
    raw = load_dataset("nyu-mll/glue", "sst2", split="validation")
    return Dataset.from_dict(
        {
            "uid": list(range(len(raw))),
            "text": raw["sentence"],
            "label": raw["label"],
        }
    )


def build_imdb_test(n: int = 2000, seed: int = 0) -> Dataset:
    raw = load_dataset("stanfordnlp/imdb", split="test")
    labels = np.array(raw["label"])
    rng = np.random.default_rng(seed)
    per_class = n // 2
    pos = rng.choice(np.where(labels == 1)[0], per_class, replace=False)
    neg = rng.choice(np.where(labels == 0)[0], per_class, replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    sub = raw.select(idx.tolist())
    # IMDb reviews contain HTML line breaks; clean them so the model sees plain text.
    texts = [t.replace("<br />", " ").replace("<br/>", " ") for t in sub["text"]]
    return Dataset.from_dict(
        {"uid": [int(i) for i in idx], "text": texts, "label": sub["label"]}
    )


def get_datasets(cache_dir: str, n_train: int, noise_rate: float,
                 data_seed: int = 0, n_imdb: int = 2000):
    """Build (or load cached) datasets. Cached so all runs share identical data."""
    tag = f"sst2_n{n_train}_noise{noise_rate:.2f}_seed{data_seed}"
    train_path = os.path.join(cache_dir, tag)
    sst2_path = os.path.join(cache_dir, "sst2_validation")
    imdb_path = os.path.join(cache_dir, f"imdb_test_{n_imdb}")

    if not os.path.exists(train_path):
        build_train_set(n_train, noise_rate, data_seed).save_to_disk(train_path)
    if not os.path.exists(sst2_path):
        build_sst2_test().save_to_disk(sst2_path)
    if not os.path.exists(imdb_path):
        build_imdb_test(n_imdb).save_to_disk(imdb_path)

    return load_from_disk(train_path), load_from_disk(sst2_path), load_from_disk(imdb_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build and inspect the datasets.")
    p.add_argument("--cache_dir", default="data")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--data_seed", type=int, default=0)
    a = p.parse_args()

    train, sst2, imdb = get_datasets(a.cache_dir, a.n_train, a.noise, a.data_seed)
    corr = np.array(train["is_corrupted"])
    agree = np.mean(np.array(train["label"]) == np.array(train["true_label"]))
    print(f"train: {len(train)} examples, {corr.sum()} corrupted ({corr.mean():.1%}), "
          f"label agreement with gold = {agree:.1%}")
    print(f"train label balance (noisy): {np.mean(train['label']):.3f} positive")
    print(f"SST-2 test: {len(sst2)}  |  IMDb test: {len(imdb)}")
    print("\nA few corrupted examples:")
    for row in train.filter(lambda r: r["is_corrupted"]).select(range(5)):
        print(f"  gold={row['true_label']} shown={row['label']} :: {row['text'][:90]}")
