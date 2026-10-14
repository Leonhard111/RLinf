# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Unit checks for the opt-in action-expert compilation hook."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.pi05_latency.runtime import (
    clone_cuda_graph_outputs,
    compile_and_warm_model,
)


def test_compile_and_warm_suffix(monkeypatch):
    """Compile only the suffix and perform three model warmup calls."""
    calls = []

    def original_suffix():
        return None

    def compiled_suffix():
        return None

    model = SimpleNamespace(model=SimpleNamespace(run_suffix=original_suffix))

    def predict_action_batch(model_input):
        calls.append(model_input)

    model.predict_action_batch = predict_action_batch

    def fake_compile(fn, *, mode):
        assert fn is original_suffix
        assert mode == "reduce-overhead"
        return compiled_suffix

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    payload = {"states": object()}
    elapsed = compile_and_warm_model(model, payload)
    assert model.model.run_suffix is compiled_suffix
    assert calls == [payload] * 3
    assert elapsed >= 0


def test_clone_cuda_graph_outputs():
    source = torch.tensor([1.0])
    copied = clone_cuda_graph_outputs((source, {"cache": [source]}, None))
    assert copied[0].data_ptr() != source.data_ptr()
    assert copied[1]["cache"][0].data_ptr() != source.data_ptr()
    assert copied[0].item() == source.item()
    assert copied[2] is None


def test_compile_and_warm_prefix_suffix(monkeypatch):
    source = torch.tensor([2.0])
    calls = []
    marks = []
    observed = []

    def original_suffix():
        return None

    def original_prefix(_observation):
        return (source, {"cache": [source]})

    model = SimpleNamespace(
        model=SimpleNamespace(
            run_suffix=original_suffix,
            build_prefix_cache=original_prefix,
        )
    )

    def predict_action_batch(model_input):
        observed.append(model.model.build_prefix_cache(model_input))

    model.predict_action_batch = predict_action_batch

    def fake_compile(fn, *, mode):
        calls.append((fn, mode))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(1)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    compile_and_warm_model(model, {"image": source}, compile_prefix=True)
    assert calls == [
        (original_suffix, "reduce-overhead"),
        (original_prefix, "reduce-overhead"),
    ]
    assert len(marks) == len(observed) == 3
    assert all(item[0].data_ptr() != source.data_ptr() for item in observed)
    assert all(item[1]["cache"][0].data_ptr() != source.data_ptr() for item in observed)
