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
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.layers = torch.nn.ModuleDict()
        for layer_idx in range(args.layers):
            self.layers[str(layer_idx)] = TransformerBlock(layer_idx, args)

        self.norm = nn.RMSNorm(args.dim)
        self.output = nn.Linear(
            args.dim, 
            args.vocab_size, 
            dtype=torch.get_default_dtype(),
            bias=False,
        )


    def init_weights(
        self,
        init_std:float | None = None,
        buffer_device:torch.device | None = None
    ):
        buffer_device = buffer_device or  self.freqs_cis.device

