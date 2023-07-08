# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from datasets import load_dataset, concatenate_datasets  # huggingface datasets
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer

import torch  # make sure to import torch

def prepare(
    destination_path: Path = Path("data/code"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    seed: int = 42,
    test_size: Union[float, int, None] = 0.0005,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    num_proc = os.cpu_count() // 2
    num_proc_load_dataset = num_proc
    dataset = load_dataset("codeparrot/codeparrot-clean", num_proc=num_proc_load_dataset)

    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = tokenizer.encode(example["content"]).tolist()
        ids.append(tokenizer.eos_id)
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(process, remove_columns=["content"], desc="tokenizing the splits", num_proc=num_proc)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.int64)
        filename = destination_path / f"{split}.pt"  # change the extension to .pt
        total_batches = 1024

        big_list = []  # Create empty list

        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            big_list.extend(arr_batch.tolist())  # Extend list with new ids

        # Save data to the file
        torch.save(big_list, str(filename))

if __name__ == "__main__":
    from jsonargparse.cli import CLI

    CLI(prepare)
