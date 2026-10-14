# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Experimental SDPA replacement for RLinf's multi-expert Gemma attention.

This is installed only by the offline benchmark. No training or deployment
module imports it by default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma


def sdpa_forward(self, xs, positions, attn_mask, kv_cache=None):
    """Match the original projection/cache contract using fused CUDA SDPA."""
    q_parts, k_parts, v_parts = [], [], []
    for index, x in enumerate(xs):
        if x is None:
            continue
        batch, length, _ = x.shape
        if self.k_proj[index] is None:
            qkv = self.q_proj[index](x).reshape(
                batch, length, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
        else:
            q = self.q_proj[index](x).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            k = self.k_proj[index](x).reshape(
                batch, length, self.num_kv_heads, self.head_dim
            )
            v = self.v_proj[index](x).reshape(
                batch, length, self.num_kv_heads, self.head_dim
            )
        q_parts.append(q)
        k_parts.append(k)
        v_parts.append(v)

    q = gemma._apply_rope(torch.cat(q_parts, dim=1), positions=positions)
    k = gemma._apply_rope(torch.cat(k_parts, dim=1), positions=positions)
    v = torch.cat(v_parts, dim=1)
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        k = torch.cat([cache_k, k], dim=1)
        v = torch.cat([cache_v, v], dim=1)
    new_kv_cache = (k, v)

    encoded = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask[:, 0:1].bool(),
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=self.num_heads != self.num_kv_heads,
    ).transpose(1, 2)

    outputs = []
    start = 0
    for index, x in enumerate(xs):
        if x is None:
            outputs.append(None)
            continue
        end = start + x.shape[1]
        outputs.append(
            self.o_proj[index](encoded[:, start:end].reshape(batch, x.shape[1], -1))
        )
        start = end
    return outputs, new_kv_cache


def install() -> None:
    """Patch the current benchmark process, leaving repository code unchanged."""
    gemma.Attention.forward = sdpa_forward
