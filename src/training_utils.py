from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from typing import Tuple
from src.dataset import CzechCorpusDataset, create_train_val_split
import torch
from src.model import CzechGPTModel, CzechGPTConfig

import math
from src.muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam
from typing import List

from tqdm.auto import tqdm
import re


def get_dataloaders(args) -> Tuple[DataLoader, DataLoader, Tokenizer]:
    
    train_path, val_path = create_train_val_split(args.corpus_path, args.val_split_ratio)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    train_dataset = CzechCorpusDataset(args.tokenizer_path, train_path, args.context_length)
    val_dataset = CzechCorpusDataset(args.tokenizer_path, val_path, args.context_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader, tokenizer


def create_model(args, tokenizer, device):
    model_config = CzechGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
        tie_word_embeddings=not args.no_tying,
    )
    model = CzechGPTModel(model_config)
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    
    if not args.no_compile:
        print("Compiling model... (This may take a minute)")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully.")

    return model


def get_lr(it, args):
    if it < args.warmup_steps:
        return args.muon_lr * it / args.warmup_steps,  args.adam_lr * it / args.warmup_steps
    if it > args.max_steps:
        return 0.0, 0.0 
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.muon_lr * coeff, args.adam_lr * coeff


def create_optimizer(model, args):
    hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
    hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
    nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
    
    use_muon = not args.no_muon
    param_groups = [
        dict(params=hidden_weights, use_muon=use_muon,
             lr=args.muon_lr, weight_decay=args.weight_decay),
        dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
             lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
    ]

    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer


@torch.no_grad()
def estimate_loss(model, val_loader: DataLoader, device: str, ctx) -> float:
    model.eval()
    losses = []
    pbar = tqdm(val_loader, desc="Estimating validation loss", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with ctx:
            outputs = model(x, labels=y)
        losses.append(outputs.loss.item())
    model.train()
    return torch.tensor(losses).mean().item()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-záčďéěíňóřšťúůýž0-9\s.,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@torch.no_grad()
def generate_samples(
    model, tokenizer: Tokenizer, device: str, prompts: List[str], args
) -> List[str]:
    model.eval()
    outputs = []
    pad_token_id = tokenizer.get_vocab().get("[PAD]", 0)

    for prompt in prompts:
        prompt = clean_text(prompt)
        token_ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_k=args.gen_top_k,
            do_sample=True,
            pad_token_id=pad_token_id
        )
        full_output = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        outputs.append(full_output)
        
    model.train()
    return outputs
