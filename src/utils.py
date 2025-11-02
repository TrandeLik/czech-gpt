import random
import numpy as np
import torch
from contextlib import nullcontext


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_torch_context(device: str):
    is_cuda = (device == 'cuda')
    is_bf16_supported = (is_cuda and torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    if is_cuda and is_bf16_supported:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()
