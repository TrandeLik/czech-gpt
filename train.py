import os
import argparse
import math
import time
from contextlib import nullcontext

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
import comet_ml
import re

from model import CzechGPTModel, CzechGPTConfig
from dataset import CzechCorpusDataset, create_train_val_split
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam


def get_args():
    parser = argparse.ArgumentParser(description="Pre-train a small Czech GPT model.")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers_cs/bpe-vocab_59.json", help="Path to the trained tokenizer.")
    parser.add_argument("--corpus_path", type=str, default="ces_wikipedia_2021_1M/cleaned_czech_corpus_for_training.txt", help="Path to the clean corpus.")
    parser.add_argument("--val_split_ratio", type=float, default=0.0005, help="Percentage of data to use for validation.")
    
    parser.add_argument("--context_length", type=int, default=256, help="Input sequence length.")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension.")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size per device.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps. Effective batch size = batch_size * grad_accum_steps.")
    parser.add_argument("--max_steps", type=int, default=10_000, help="Total number of training steps.")

    parser.add_argument("--no_muon", action="store_true", help="Disable Muon")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="LR for hidden weights (Muon group).")
    parser.add_argument("--adam_lr", type=float, default=3e-4, help="LR for biases, gains, and non-hidden params (Adam group).")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=300, help="Number of warmup steps for LR scheduler.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile for debugging.")
    
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluate on validation set every N steps.")
    parser.add_argument("--experiment_suffix", type=str, default="", help="Suffix for checkpoint dir.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training metrics every N steps.")
    parser.add_argument("--comet_project", type=str, default="czech-gpt-pretrain", help="Comet ML project name.")
    parser.add_argument("--comet_workspace", type=str, default=None, help="Comet ML workspace name (optional).")

    parser.add_argument("--gen_max_new_tokens", type=int, default=60, help="Max new tokens to generate during validation.")
    parser.add_argument("--gen_temperature", type=float, default=0.8, help="Temperature for sampling during generation.")
    parser.add_argument("--gen_top_k", type=int, default=40, help="Top-k sampling during generation.")

    return parser.parse_args()

def get_lr(it, args):
    if it < args.warmup_steps:
        return args.muon_lr * it / args.warmup_steps,  args.adam_lr * it / args.warmup_steps
    if it > args.max_steps:
        return 0.0 
    decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.muon_lr * coeff, args.adam_lr * coeff


@torch.no_grad()
def estimate_loss(model, val_loader, device, ctx):
    model.eval()
    losses = []
    pbar = tqdm(val_loader, desc="Estimating validation loss", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with ctx:
            outputs = model(x, labels=y)
        loss = outputs.loss
        losses.append(loss.item())
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
def generate_samples(model, tokenizer, device, prompts, max_new_tokens, temperature, top_k):
    model.eval()
    outputs = []
    
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.get_vocab().get("[PAD]", 0)
        )
        full_output = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        outputs.append(full_output)
        
    model.train()
    return outputs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    experiment = comet_ml.start(
        project_name=args.comet_project,
        workspace=args.comet_workspace
    )
    experiment.log_parameters(vars(args))
    
    train_path, val_path = create_train_val_split(args.corpus_path, args.val_split_ratio)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    train_dataset = CzechCorpusDataset(args.tokenizer_path, train_path, args.context_length)
    val_dataset = CzechCorpusDataset(args.tokenizer_path, val_path, args.context_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    model_config = CzechGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    )
    model = CzechGPTModel(model_config)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    
    if not args.no_compile:
        model = torch.compile(model, mode="reduce-overhead") 

    num_params = model.get_num_params()
    chinchilla_tokens = 20 * num_params
    print(f"Model has {num_params:,} parameters.")
    print(f"Chinchilla optimal tokens: {chinchilla_tokens:,}")
    experiment.log_metric("num_params", num_params)
    experiment.log_metric("chinchilla_tokens_target", chinchilla_tokens)

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
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == 'cuda')) if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else nullcontext()


    step = 0
    tokens_processed = 0
    t0 = time.time()
    
    data_iter = iter(train_loader)

    prompts = [
        "Včera jsem byl v obchodě a koupil jsem si",
        "Jablko je ovoce. Mrkev je zelenina. Kuře je",
        "Jak se do lesa volá, tak se z lesa",
        
        "Otázka: Jakou barvu má nebe za slunečného dne?\nOdpověď:",

        "Věta: Ten film byl naprosto úžasný! Sentiment: pozitivní\n---\nVěta: Jídlo v restauraci bylo studené a bez chuti. Sentiment: negativní\n---\nVěta: Po dlouhém dni v práci jsem konečně doma. Sentiment:",

        "Otázka: Jaké je hlavní město České republiky?\nOdpověď: Praha.\n---\nOtázka: Kdo napsal Babičku?\nOdpověď:",
        "Báseň o Praze:\nKámen a čas, Vltava zpívá,\nvěž starých snů se k nebi dívá.\n---\nBáseň o lese:\n",
    ]
    
    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        torch.compiler.cudagraph_mark_step_begin()
        x, y = x.to(device), y.to(device)
        
        muon_lr, adam_lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            if param_group['use_muon']:
                param_group['lr'] = muon_lr
            else:
                param_group['lr'] = adam_lr

        for micro_step in range(args.grad_accum_steps):
            with ctx:
                outputs = model(x, labels=y)
                loss = outputs.loss
                loss = loss / args.grad_accum_steps
            
            scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_processed += (args.batch_size * args.context_length)
        
        if step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (args.batch_size * args.grad_accum_steps * args.context_length) / dt
            print(f"Step {step}/{args.max_steps} | Loss: {loss.item()*args.grad_accum_steps:.4f} | Adam LR: {adam_lr:.2e} | Muon LR: {muon_lr:.2e} | Tokens/s: {tokens_per_sec:.0f}")
            experiment.log_metrics({
                "train_loss": loss.item() * args.grad_accum_steps,
                "adam_learning_rate": adam_lr,
                "muon_learning_rate": muon_lr,
                "tokens_per_sec": tokens_per_sec,
                "tokens_processed": tokens_processed,
            }, step=step)

        if step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, ctx)
            perplexity = math.exp(val_loss)
            print(f"--- Validation --- Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f}")
            experiment.log_metrics({
                "val_loss": val_loss,
                "val_perplexity": perplexity,
            }, step=step)
            print("--- Generating Samples ---")
            generated_texts = generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_k=args.gen_top_k
            )
            print(f"Sample 0:\n{generated_texts[0]}")

            table_data = []
            for prompt, generation in zip(prompts, generated_texts):
                table_data.append([step, prompt, generation])

            experiment.log_table(
                filename="generation_samples.csv",
                tabular_data=table_data,
                headers=["Step", "Prompt", "Generation"],
                step=step
            )
            print("Logged generation samples to a Comet ML table.")

            checkpoint_dir = f"checkpoint_{args.experiment_suffix}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir, safe_serialization=False)
            tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))


if __name__ == "__main__":
    args = get_args()
    main(args)
