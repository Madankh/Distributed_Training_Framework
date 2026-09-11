import math
from typing import Optional
from torch import nn 

from TorchDist.model.model.attention import (ScaledDotProductAttentionWrapper)
from TorchDist.model.model_args import DeepSeekV3Args
from TorchDist.model.moe import FeedForward, MoE
from TorchDist.model.rope import apply_rotary_emb, precompute_freqs_cis

class DeepSeekV3model(nn.Module):
    def __init__(self, args:DeepSeekV3Args):
        super().__init__()
        self.model_args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)