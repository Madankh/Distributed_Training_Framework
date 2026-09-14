import math
from typing import Optional
from torch import nn 

from TorchDist.model.model.attention import (ScaledDotProductAttentionWrapper)
from TorchDist.model.model_args import DeepSeekV3Args
from TorchDist.model.moe import FeedForward, MoE
from TorchDist.model.rope import apply_rotary_emb, precompute_freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self, layer_id:int, args:DeepSeekV3Args):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim,eps=args.norm_eps)

        self.moe_enabled = layer_id >= args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                args.moe.args,
                dim=args.dim,
                hidden_dim=args.moe_inter_dim
            )
        else:
            self.feed_forward = FeedForward(args.dim, args.inter_dim)
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x:torch.Tensor, freqs_cis:torch.Tensor):
        x = x + self.attention(self.attention_norm(x),freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self,
        init_std:float | None = None,
        buffer_device:torch.device | None = None
    ):
     if buffer_device is None:
        raise ValueError("buffer_device must be provided for TransformerBlock weight init")
     for norm in (self.attention_norm, self.ffn_norm):
        norm.reset_parameters()
     self.attention.init_weights(self.weight_init_std)
     if self.moe_enabled:
        self.moe.init_weights(
            init_std=self.weight_init_std,buffer_device=buffer_device
        )
     else:
        self.feed_forward.init_weights(self.weight_init_std)


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
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std,buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight, 
                mean=0.0,
                std=final_out_std,
                a = -cutoff_factor * final_out_std,
                b = cutoff_factor * final_out_std
            )

    def forward(self, tokens:torch.Tensor):
        h = self.tok_embeddings(tokens) if self.tok_embedding is not None else tokens
        for layer_idx in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
