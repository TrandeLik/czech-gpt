import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin


class CzechGPTConfig(PretrainedConfig):
    model_type = "czech-gpt"
    def __init__(
            self, 
            vocab_size=32000, 
            context_length=256, 
            n_embd=384,      
            n_layer=6, 
            n_head=6, 
            dropout=0.1, 
            tie_word_embeddings=True, 
            **kwargs
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout


class QKNorm(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.q_norm = nn.LayerNorm(head_size, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_size, eps=1e-6)

    def forward(self, q, k): 
        return self.q_norm(q), self.k_norm(k)


class RoPE(nn.Module):
    def __init__(self, head_size, context_length):
        super().__init__()
        theta = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(context_length).float()
        freqs = torch.outer(positions, theta)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(1)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def _apply_rope(self, x, freqs_complex):
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        rotated_x_complex = x_complex * freqs_complex
        rotated_x_reshaped = torch.view_as_real(rotated_x_complex)
        rotated_x = rotated_x_reshaped.flatten(-2)
        return rotated_x.type_as(x)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        freqs_complex = self.freqs_complex[:, :, :seq_len, :]
        q_rotated = self._apply_rope(q, freqs_complex)
        k_rotated = self._apply_rope(k, freqs_complex)
        return q_rotated, k_rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.head_size = config.n_embd // self.n_head
        self.qk_norm = QKNorm(self.head_size)
        self.rope = RoPE(self.head_size, config.context_length)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, k = self.rope(q, k)
        q, k = self.qk_norm(q, k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = torch.pow(F.relu(x), 2)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__(); self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CzechGPTModel(PreTrainedModel, GenerationMixin):
    config_class = CzechGPTConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.main_input_name = "input_ids"

        self.embed = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
        ))
        
        self.body = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.embed.wte.weight = self.head.weight # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
    ):
        B, T = input_ids.size()
        assert T <= self.config.context_length, f"Sequence length {T} exceeds model context length {self.config.context_length}"

        tok_emb = self.embed.wte(input_ids)
        x = self.embed.drop(tok_emb)
        
        for block in self.body.h:
            x = block(x)
        x = self.body.ln_f(x)
        
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
