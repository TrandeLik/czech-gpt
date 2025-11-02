import time
import comet_ml


class ExperimentLogger:
    def __init__(self, args):
        self.experiment = comet_ml.start(
            project_name=args.comet_project,
            workspace=args.comet_workspace
        )
        self.experiment.log_parameters(vars(args))
        self.t0 = time.time()

    def log_model_info(self, model):
        num_params = model.get_num_params()
        chinchilla_tokens = 20 * num_params
        print(f"Model has {num_params:,} parameters.")
        print(f"Chinchilla optimal tokens for this model: {chinchilla_tokens:,}")
        self.experiment.log_metrics({
            "num_params": num_params,
            "chinchilla_tokens_target": chinchilla_tokens,
        }, step=0)

    def log_train_step(self, step, max_steps, loss, lrs, tokens_processed, tokens_in_step):
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1
        tokens_per_sec = tokens_in_step / dt
        print(
            f"Step {step}/{max_steps} | Loss: {loss:.4f} | "
            f"Adam LR: {lrs[1]:.2e} | Muon LR: {lrs[0]:.2e} | "
            f"Tokens/s: {tokens_per_sec:.0f}"
        )
        self.experiment.log_metrics({
            "train_loss": loss,
            "adam_learning_rate": lrs[1],
            "muon_learning_rate": lrs[0],
            "tokens_per_sec": tokens_per_sec,
            "tokens_processed": tokens_processed,
        }, step=step)

    def log_eval_step(self, step, val_loss, perplexity, prompts, generations):
        print(f"--- Validation --- Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f}")
        print("--- Generating Samples ---")
        print(f"Sample 0:\n{generations[0]}\n")
        self.experiment.log_metrics({
            "val_loss": val_loss,
            "val_perplexity": perplexity,
        }, step=step)

        table_data = [[step, p, g] for p, g in zip(prompts, generations)]
        self.experiment.log_table(
            filename="generation_samples.csv",
            tabular_data=table_data,
            headers=["Step", "Prompt", "Generation"],
            step=step
        )
