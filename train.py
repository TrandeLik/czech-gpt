from src.training_utils import get_dataloaders, create_model, create_optimizer
from src.trainer import Trainer
from src.utils import set_seed, get_device
from src.logging_utils import ExperimentLogger
import argparse

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
    parser.add_argument("--no_tying", action="store_true", help="Disable Weight Tying")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size per device.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_steps", type=int, default=10_000, help="Total number of training steps.")

    parser.add_argument("--no_muon", action="store_true", help="Disable Muon")
    parser.add_argument("--muon_lr", type=float, default=0.02, help="LR for hidden weights (Muon group).")
    parser.add_argument("--adam_lr", type=float, default=3e-4, help="LR for biases, gains, and non-hidden params (Adam group).")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=300, help="Number of warmup steps for LR scheduler.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile for debugging.")
    
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluate on validation set every N steps.")
    parser.add_argument("--experiment_suffix", type=str, default="", help="Suffix for checkpoint dir.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training metrics every N steps.")
    parser.add_argument("--comet_project", type=str, default="czech-gpt-pretrain", help="Comet ML project name.")
    parser.add_argument("--comet_workspace", type=str, default=None, help="Comet ML workspace name (optional).")

    parser.add_argument("--gen_max_new_tokens", type=int, default=60, help="Max new tokens for generation.")
    parser.add_argument("--gen_temperature", type=float, default=0.8, help="Temperature for generation.")
    parser.add_argument("--gen_top_k", type=int, default=40, help="Top-k for generation.")

    return parser.parse_args()


def main():
    args = get_args()
    set_seed(42)
    device = get_device()
    logger = ExperimentLogger(args)
    train_loader, val_loader, tokenizer = get_dataloaders(args)
    model = create_model(args, tokenizer, device)
    optimizer = create_optimizer(model, args)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        logger=logger,
        args=args
    )
    trainer.train()

if __name__ == "__main__":
    main()
