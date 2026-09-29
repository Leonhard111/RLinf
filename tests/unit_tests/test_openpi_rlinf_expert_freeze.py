# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainability contract for OpenPI RLinf RLT Stage 1."""

import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.openpi_rlinf.utils.model_builders import _build_sft_model


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)]))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention_norms = nn.ModuleList([nn.LayerNorm(8), nn.LayerNorm(8)])
        self.pre_ffw_norms = nn.ModuleList([nn.LayerNorm(8), nn.LayerNorm(8)])
        self.mlps = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.attn = _Attention()


class _LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Embedding(8, 8)
        self.layers = nn.ModuleList([_Block()])
        self.final_norms = nn.ModuleList([nn.LayerNorm(8), nn.LayerNorm(8)])


class _TinyPi0(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = nn.Linear(8, 8)
        self.llm = _LLM()
        self.action_in_proj = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.time_mlp_in = nn.Linear(8, 8)
        self.time_mlp_out = nn.Linear(8, 8)


def _make_sft_model(train_expert_only: bool):
    cfg = OmegaConf.create(
        {
            "train_expert_only": train_expert_only,
            "use_rlt": True,
            "rlt_input_dim": 8,
            "rlt_embed_dim": 8,
            "rlt_prefix_seq_len": 4,
            "rlt_num_layers": 1,
            "rlt_num_heads": 2,
        }
    )
    return _build_sft_model(cfg, _TinyPi0(), num_steps=1, action_env_dim=8)


def test_rlt_stage1_freezes_vlm_but_trains_action_expert_and_token():
    model = _make_sft_model(train_expert_only=True)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]

    assert any(name.startswith("model.img.") for name in frozen)
    assert any(name.startswith("model.llm.embedder.") for name in frozen)
    assert any("model.llm.layers.0.attn.q_proj.0." in name for name in frozen)
    assert any("model.llm.layers.0.mlps.0." in name for name in frozen)
    assert not any("model.llm.layers.0.attn.q_proj.1." in name for name in frozen)
    assert any("model.llm.layers.0.attn.q_proj.1." in name for name in trainable)
    assert any(name.startswith("model.action_out_proj.") for name in trainable)
    assert any(name.startswith("rlt_module.") for name in trainable)

    rlt_loss, _ = model._rlt_forward(
        torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)
    )
    expert_loss = model.model.llm.layers[0].mlps[1](torch.randn(1, 8)).sum()
    action_loss = model.model.action_out_proj(torch.randn(1, 8)).sum()
    (rlt_loss + expert_loss + action_loss).backward()

    assert all(p.grad is None for name, p in model.named_parameters() if name in frozen)
    assert any(
        p.grad is not None
        for name, p in model.named_parameters()
        if name.startswith("rlt_module.")
    )
    assert model.model.llm.layers[0].mlps[1].weight.grad is not None
    assert model.model.action_out_proj.weight.grad is not None
    assert model.freeze_vlm() == 0


def test_rlt_stage1_default_keeps_vlm_trainable():
    model = _make_sft_model(train_expert_only=False)
    assert all(p.requires_grad for p in model.parameters())
