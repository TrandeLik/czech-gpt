import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


def create_train_val_split(corpus_path, val_split_ratio=0.01):
    train_path = "train.txt"
    val_path = "val.txt"

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Train/validation splits already exist. Skipping creation.")
        return train_path, val_path

    print.info(f"Creating train/val split from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    split_idx = int(len(lines) * (1 - val_split_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"Created {train_path} ({len(train_lines)} lines) and {val_path} ({len(val_lines)} lines).")
    return train_path, val_path


class CzechCorpusDataset(Dataset):
    def __init__(self, tokenizer_path, file_path, context_length):
        self.context_length = context_length
        tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"Tokenizing data from {file_path}. This may take a while...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokenized_output = tokenizer.encode(text)
        self.data = torch.tensor(tokenized_output.ids, dtype=torch.long)
        print(f"Tokenization complete. Corpus has {len(self.data):,} tokens.")

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
