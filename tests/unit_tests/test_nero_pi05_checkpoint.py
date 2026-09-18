# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the Nero π0.5 strict-checkpoint verification tool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RUN_NERO = Path(__file__).resolve().parents[2] / "run" / "nero"
sys.path.insert(0, str(RUN_NERO))
from verify_pi05_checkpoint import (  # noqa: E402
    build_model_config,
    validate_action_output,
)


def test_verify_config_is_exact_nero_contract(tmp_path: Path) -> None:
    cfg = build_model_config(tmp_path / "model", tmp_path / "norm_stats.json")
    assert cfg.model_type == "openpi_rlinf"
    assert cfg.pi05 is True
    assert cfg.num_action_chunks == 50
    assert cfg.action_dim == 16
    assert cfg.openpi.config_name == "pi05_nero"
    assert cfg.openpi.model_action_dim == 32
    assert cfg.openpi.num_images_in_input == 3
    assert cfg.openpi.discrete_state_input is True


def test_validate_action_output_rejects_shape_nan_and_limits() -> None:
    valid = np.zeros((1, 50, 16), dtype=np.float32)
    valid[..., [7, 15]] = 0.05
    assert validate_action_output(valid).shape == (1, 50, 16)

    with pytest.raises(ValueError, match="shape"):
        validate_action_output(np.zeros((50, 16), dtype=np.float32))
    invalid = valid.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_action_output(invalid)
    invalid = valid.copy()
    invalid[0, 0, 0] = 99.0
    with pytest.raises(ValueError, match="hard limits"):
        validate_action_output(invalid)
