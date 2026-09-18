# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from rlinf.utils.ckpt_convertor.openpi.jax_to_openpi_rlinf import (
    _merge_lora_weight,
    _unwrap_value_wrappers,
)


def test_unwrap_value_wrappers_recursively():
    leaf = np.arange(6, dtype=np.float32).reshape(2, 3)
    tree = {"params": {"dense": {"value": leaf}, "plain": leaf + 1}}

    unwrapped = _unwrap_value_wrappers(tree)

    np.testing.assert_array_equal(unwrapped["params"]["dense"], leaf)
    np.testing.assert_array_equal(unwrapped["params"]["plain"], leaf + 1)


def test_merge_lora_weight_preserves_prefix_axes_and_scale():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
    lora_a = rng.normal(size=(2, 3, 4, 2)).astype(np.float32)
    lora_b = rng.normal(size=(2, 3, 2, 6)).astype(np.float32)

    merged = _merge_lora_weight(
        {"w": base, "lora_a": lora_a, "lora_b": lora_b},
        base_key="w",
        lora_a_key="lora_a",
        lora_b_key="lora_b",
        scale=0.5,
    )

    expected = base + 0.5 * np.matmul(lora_a, lora_b)
    np.testing.assert_allclose(merged, expected, rtol=1e-6, atol=1e-6)
    assert merged.dtype == np.float32


def test_merge_lora_weight_keeps_dense_checkpoint_unchanged():
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    merged = _merge_lora_weight(
        {"w": base},
        base_key="w",
        lora_a_key="lora_a",
        lora_b_key="lora_b",
        scale=1.0,
    )
    np.testing.assert_array_equal(merged, base)


def test_merge_lora_weight_rejects_incomplete_pair():
    with pytest.raises(ValueError, match="incomplete LoRA pair"):
        _merge_lora_weight(
            {
                "w": np.zeros((2, 2), dtype=np.float32),
                "lora_a": np.zeros((2, 1), dtype=np.float32),
            },
            base_key="w",
            lora_a_key="lora_a",
            lora_b_key="lora_b",
            scale=1.0,
        )
