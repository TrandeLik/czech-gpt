# 🇨🇿 Czech-GPT: Pre-training a Small Czech Language Model from Scratch

This project provides a complete pipeline for pre-training a small, decoder-only transformer model for the Czech language from scratch. It covers all steps from data acquisition and cleaning to tokenizer training, model pre-training, and evaluation. The codebase is built using modern and efficient techniques with PyTorch and Hugging Face.

## Project Description

The goal of this project is to build a competent small language model specifically for the Czech language. Instead of fine-tuning an existing English-centric model, we pre-train from scratch on a Czech corpus. This ensures that both the tokenizer and the model's learned representations are optimized for the unique morphological and syntactic properties of Czech.

## Data

The training data is a text corpus of the Czech Wikipedia, downloaded from the [Leipzig Corpora Collection](https://wortschatz-leipzig.de/en/download/). We use the ces_wikipedia_2021_1M dataset, which contains 1 million sentences. The raw text is heavily preprocessed to create a clean, consistent training corpus. The `data.ipynb` notebook performs the following steps:

1. **Language Detection**: Sentences not identified as Czech (cs) are removed using the langdetect library. This is crucial for creating a monolingual model.
1. **Text Cleaning**: A custom clean_text function is applied to every sentence:
   - Converts all text to lowercase.
   - Removes URLs.
   - Removes content within square brackets, which are common Wikipedia citations.
   - Removes all characters except for Czech letters (a-z and áčďéěíňóřšťúůýž), numbers, whitespace, and basic punctuation (.,!?-).
   - Normalizes all whitespace to single spaces.
1. **Final Filtering**: Very short sentences (fewer than 3 words) are removed.
1. The final output is a clean text file (cleaned_czech_corpus_for_training.txt), with one sentence per line.

## Tokenizer

A custom tokenizer is trained (`tokenizers.ipynb`) on the cleaned corpus to create a vocabulary optimized for Czech. We use the tokenizers library from Hugging Face.
- Algorithm: Byte-Pair Encoding (BPE).
- Vocabulary Size: Experiments were conducted with different vocabulary sizes. A size of 64 was chosen for a character-level tokenizer.
- Special Tokens: The tokenizer includes standard special tokens: `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.
- Pre-tokenization: Text is split by whitespace before BPE merges are performed.
- The trained tokenizer is saved as a .json file and used for both training and evaluation.

## Model Architecture

The model is a custom decoder-only (GPT-style) transformer defined in `model.py`. It is designed for efficiency and performance, incorporating several modern architectural choices:
- **Causal Attention**: Standard decoder architecture where each token can only attend to previous tokens. Implemented using the highly efficient `torch.nn.functional.scaled_dot_product_attention`, which can leverage backends like FlashAttention.
- **Rotary Positional Embeddings (RoPE)**: Instead of learned absolute positional embeddings, RoPE is used to inject relative positional information by rotating query and key vectors. This improves extrapolation to longer sequences.
- **QK Normalization**: LayerNorm is applied to the Query (Q) and Key (K) vectors before the attention score calculation. This helps stabilize training (see [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)).
- **Squared ReLU Activation**: The standard ReLU in the feed-forward network (FFN) is replaced with ReLU² (i.e., torch.pow(F.relu(x), 2)), a non-linearity that has shown empirical benefits (see [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt)).
- **Weight Tying**: The token embedding weights are tied with the final language model head weights, reducing the total number of parameters.
- **Hugging Face Compatibility**: The model inherits from PreTrainedModel and GenerationMixin, and its forward method is compatible with the Hugging Face generate() API, allowing easy use for text generation tasks.

## Training

The training script `train.py` is modular and highly configurable via command-line arguments.
1. **Optimizer**: The training script supports two optimizers: classical AdamW and modern [Muon](https://github.com/KellerJordan/Muon).
1. **Efficiency Techniques**:
  - Automatic Mixed Precision (AMP)
  - Gradient Accumulation
  - Gradient Checkpointing
  - torch.compile()
3. Logging: Training progress, including loss, perplexity, learning rates, and generated text samples, is logged to Comet ML for real-time monitoring and experiment tracking.

## Evaluation

The `benchmark.ipynb` notebook provides a comprehensive framework for both qualitative and quantitative evaluation. A critical aspect of the evaluation is that all prompts are preprocessed with the exact same cleaning function used for the training data to ensure a fair test.
- **General Text Generation**: The model's coherence and creativity are assessed with open-ended text continuation prompts.
- **In-Domain Question Answering**: The model's ability to recall facts from its training data is tested with several questions in different formulations.
- **Few-Shot Benchmarks**:
   - Factual Knowledge (Country -> Capital): Measures the model's ability to follow a few-shot pattern for a factual recall task.
   - Grammatical Skill (Singular -> Plural): Measures the model's understanding of Czech morphology by tasking it with generating the plural form of given nouns.

## Experiments

All experiments were logged and are available for review on Comet ML.
Comet ML Project Link: https://www.comet.com/trandelik/czech-gpt-pretrain/view/new/panels

### Model Summary

| Tying | Vocab Size |	Embedding Dim |	Heads 	|	Layers | Optimizer	 | Parameters count | Number of Chinchillas | Validation Perplexity |	Checkpoint Link |
| ----- | ---------- | ------------- | ---------- | ------- | ------------ | ---------------- | --------------------- | ------ | ----- |
| Yes | 64 | 768 | 12 | 12 | Muon | 85,111,296 | 0.025 | 5.918 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed768_muon10k) |
| Yes | 64 | 768 | 12 | 12 | Muon | 85,111,296 | 0.05 | 5.449 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed768_muon20k) |
| Yes | 64 | 384 | 12 | 12 | Muon | 21,321,984 | 0.22 | 5.751 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed384_muon20k) |
| Yes | 64 | 384 | 12 | 12 | Muon | 21,321,984 | 0.55 | 5.254 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed384_muon50k) |
| Yes | 64 | 384 | 12 | 12 | Muon | 21,321,984 | 1.00 | 4.992 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed384_muon100k) |
| No | 64 | 384 | 12 | 12 | Muon | 21,321,984 | 1.14 | **4.944** | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed384_muon100k) |
| Yes | 64 | 384 | 12 | 12 | AdamW | 21,321,984 | 0.55 | 5.452 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed384_adamw50k) |
| Yes | 64 | 64 | 4 | 12 | Muon | 604,800 | 6.77 | 7.972 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed64_muon20k) |
| Yes | 64 | 64 | 4 | 4 | Muon | 204,416 | 20.03 | 10.28 | [HF](https://huggingface.co/TrandeLik/czech_gpt_vocab_64_embed64_layers4_muon50k) |

## Reproducibility

- Setup Environment: create environment from environment.yml (e.g., using `micromamba`)
- Prepare Data: Run the `data.ipynb notebook` to download and preprocess the Wikipedia corpus.
- Train Tokenizer: Run the `tokenizers.ipynb` notebook to train your custom tokenizer. Ensure the path in `train.py` points to the correct tokenizer file.
- Set Up Logging: login into Comet ML
- Start Pre-training: Run the training script. You can configure all hyperparameters via command-line arguments or run `run.sh` script to reproduce reported results
- Evaluate Model: After training, update the `CHECKPOINT_PATH` in benchmark.ipynb and run the cells to evaluate your trained model.

## Conclusion

- Muon is good even in this simple task
- Chinchilla scaling is good even in this simple task