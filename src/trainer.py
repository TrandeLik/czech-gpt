import os
import math
import torch
from contextlib import nullcontext

from src.training_utils import get_lr
from src.training_utils import estimate_loss, generate_samples
from src.utils import get_torch_context, get_device

PROMPTS = [
    "Včera jsem byl v obchodě a koupil jsem si",
    "Jablko je ovoce. Mrkev je zelenina. Kuře je",
    "Jak se do lesa volá, tak se z lesa",
    "Otázka: Jakou barvu má nebe za slunečného dne?\nOdpověď:",
    "Věta: Ten film byl naprosto úžasný! Sentiment: pozitivní\n---\nVěta: Jídlo v restauraci bylo studené a bez chuti. Sentiment: negativní\n---\nVěta: Po dlouhém dni v práci jsem konečně doma. Sentiment:",
    "Otázka: Jaké je hlavní město České republiky?\nOdpověď: Praha.\n---\nOtázka: Kdo napsal Babičku?\nOdpověď:",
    "Báseň o Praze:\nKámen a čas, Vltava zpívá,\nvěž starých snů se k nebi dívá.\n---\nBáseň o lese:\n",
]


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, tokenizer, logger, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args

        self.device = get_device()
        self.ctx = get_torch_context(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))
        
        self.step = 0
        self.tokens_processed = 0

    def _run_evaluation(self):
        val_loss = estimate_loss(self.model, self.val_loader, self.device, self.ctx)
        perplexity = math.exp(val_loss)
        
        generated_texts = generate_samples(self.model, self.tokenizer, self.device, PROMPTS, self.args)
        
        self.logger.log_eval_step(self.step, val_loss, perplexity, PROMPTS, generated_texts)
        self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_dir = f"checkpoint_{self.args.experiment_suffix}" if self.args.experiment_suffix else "checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model_to_save.save_pretrained(checkpoint_dir, safe_serialization=False)
        self.tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
        print(f"Checkpoint saved to {checkpoint_dir}")

    def train(self):
        self.logger.log_model_info(self.model)
        data_iter = iter(self.train_loader)

        while self.step < self.args.max_steps:
            lrs = get_lr(self.step, self.args)
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lrs[i]

            for _ in range(self.args.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)
                with self.ctx:
                    outputs = self.model(x, labels=y)
                    loss = outputs.loss / self.args.grad_accum_steps
                
                self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            tokens_in_step = self.args.batch_size * self.args.grad_accum_steps * self.args.context_length
            self.tokens_processed += tokens_in_step
            if self.step % self.args.log_interval == 0:
                self.logger.log_train_step(
                    self.step, self.args.max_steps, loss.item() * self.args.grad_accum_steps,
                    lrs, self.tokens_processed, tokens_in_step
                )
            if self.step % self.args.eval_interval == 0:
                self._run_evaluation()
