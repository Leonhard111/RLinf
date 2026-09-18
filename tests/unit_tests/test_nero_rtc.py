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

from __future__ import annotations

import torch

from rlinf.models.embodiment.openpi_rlinf.rtc import (
    RTCModelContext,
    align_previous_chunk,
    guided_velocity,
    prefix_weights,
)


def test_prefix_weights_match_reference_example() -> None:
    weights = prefix_weights(
        2,
        6,
        10,
        "linear",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]),
    )


def test_previous_chunk_is_shifted_to_request_time() -> None:
    previous = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    aligned = align_previous_chunk(previous, 2)
    torch.testing.assert_close(aligned[:, :3], previous[:, 2:])
    torch.testing.assert_close(aligned[:, 3:], torch.zeros(1, 2, 2))


def test_context_rounds_integer_fields_after_tensor_transport() -> None:
    value = torch.tensor(
        [[1.0, 4.0, 3.0, 120.0, 30.0, 17.0, 50.0, 0.0, 5.0]],
        dtype=torch.float64,
    )
    context = RTCModelContext.from_tensor(value)
    assert context is not None
    assert context.enabled
    assert context.previous_generation_id == 3
    assert context.prefix_attention_schedule == "exp"


def test_guided_velocity_uses_denoiser_vjp() -> None:
    x_t = torch.full((1, 5, 2), 0.25)
    previous = torch.zeros_like(x_t)
    base = 2.0 * x_t
    guided = guided_velocity(
        x_t,
        previous,
        inference_delay_steps=1,
        prefix_attention_horizon=3,
        time=0.8,
        schedule="linear",
        max_guidance_weight=5.0,
        denoise_fn=lambda value: 2.0 * value,
    )
    assert guided.shape == x_t.shape
    assert torch.isfinite(guided).all()
    assert not torch.equal(guided, base)
