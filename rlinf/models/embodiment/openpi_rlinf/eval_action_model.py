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

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.rtc import (
    RTCModelContext,
    align_previous_chunk,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)
from rlinf.utils.latency_logger import get_latency_logger


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


class OpenPiPytorchEvalActionModel(OpenPiPytorchActionModel):
    """Eval-capable wrapper around the vendored ``Pi0`` model.

    Drives observation construction through the upstream ``openpi.transforms``
    pipeline: the factory calls :meth:`setup_wrappers` with the composed
    input/output transform lists from :func:`get_openpi_config`, then
    :meth:`predict_action_batch` routes ``env_obs`` through
    :meth:`_repack_env_obs` → :meth:`input_transform` →
    :meth:`_observation_dict_to_device` → :meth:`Pi0.sample_actions` →
    :meth:`output_transform`.

    The RL subclass inherits this eval path unchanged and adds the PPO
    chain-collecting SDE sampler on top.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )
        # ``action_chunk`` slices the env-action subspace from the model output
        # in :meth:`output_transform`.
        self.action_chunk = action_chunk
        # ``config_name`` is the openpi TrainConfig key (e.g. ``pi05_behavior``)
        # used by :meth:`_repack_env_obs` to switch the env→``observation/*``
        # repack rules (currently only the ``calvin`` split-state layout).
        self.config_name = config_name
        # Optional subset of the raw env state dim (openpi ``state_indices``).
        # ``None`` (the BEHAVIOR default) is an identity passthrough.
        self.state_indices = list(state_indices) if state_indices else None

        # openpi.transforms pipeline state (installed by :meth:`setup_wrappers`).
        self._input_transform_fn = None
        self._output_transform_fn = None
        self._latency_profile = os.environ.get("RLINF_PI05_LATENCY_PROFILE") == "1"
        self._latency_logger = get_latency_logger("pi05_inference")
        self._rtc_previous_model_actions: torch.Tensor | None = None
        self._rtc_previous_generation_id: int | None = None

    # -------------------------------------------------------- transforms glue

    def setup_wrappers(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines.

        ``transforms`` is the list passed to the openpi-side ``compose`` —
        typically ``BehaviorInputs → Normalize(norm_stats) → ModelTransformFactory(model_config)``
        (the last stage carries the auto-downloading PaliGemma tokenizer).
        ``output_transforms`` is the matching reverse pipeline used to turn
        sampled model actions back into env-frame actions.
        """
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)

    def _ensure_wrappers(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_wrappers(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity).

        Mirrors ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.
        _select_configured_state``: ``state_indices=None`` (BEHAVIOR) is an
        identity passthrough; a configured index list fancy-indexes the last
        dim so future non-BEHAVIOR envs can reuse the generic repack.
        """
        indices = self.state_indices
        if not indices:
            return states

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def _repack_env_obs(self, env_obs: dict) -> dict:
        """Map the env's observation dict to the ``observation/*`` keys the
        openpi pipeline expects.

        Generic adapter mirroring
        ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.obs_processor``:
        it reads the standardized RLinf env keys (``states`` / ``main_images`` /
        ``task_descriptions`` and the optional ``wrist_images`` /
        ``extra_view_images``) and repacks them into openpi's ``observation/*``
        convention. All real per-env logic lives in the config-driven
        ``data_transforms.inputs`` (selected by ``config_name``), not here; the
        only per-env branch is the ``calvin`` split-state layout. Optional
        camera views are added only when the env actually provides them
        (BEHAVIOR, for instance, emits no ``extra_view_images`` key).
        """
        env_states = self._select_configured_state(env_obs["states"])
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config_name:
            processed_obs["observation/state_ee_pos"] = env_states[:, :3]
            processed_obs["observation/state_ee_rot"] = env_states[:, 3:6]
            processed_obs["observation/state_gripper"] = env_states[:, 6:7]
        else:
            processed_obs["observation/state"] = env_states
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is not None:
            processed_obs["observation/wrist_image"] = wrist_images
        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            processed_obs["observation/extra_view_image"] = extra_view_images
        return processed_obs

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        """Apply the openpi input pipeline per-sample then recombine into a batched dict.

        Two modes:

        * Rollout (``"prompt"`` key present) — runs the full pipeline including
          prompt tokenization; result has ``image``/``image_mask``/``state``/
          ``tokenized_prompt``/``tokenized_prompt_mask`` keys.
        * Train recompute (no ``"prompt"``; only ``observation/*`` and the cached
          ``tokenized_prompt`` keys present) — re-runs the pipeline using the
          cached tokens rather than re-tokenising every micro-batch.
        """
        self._ensure_wrappers()
        inputs = tree_map(lambda x: x, obs)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {k: inputs[k] for k in inputs.keys() if "/" in k}

        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0)
                        if isinstance(x, np.ndarray) and x.ndim == 3
                        else x
                    ),
                    sample,
                )
            if first_process:
                prompts = obs["prompt"]
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()
                sample["prompt"] = prompts[i]
            else:
                # Pipeline still runs Tokenize, but the cached tokens below
                # overwrite its output — placeholder text is fine.
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)

        if len(batch_samples) == 1:
            transformed = [self._input_transform_fn(batch_samples[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
                transformed = list(ex.map(self._input_transform_fn, batch_samples))

        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if not first_process:
            recombined["tokenized_prompt"] = obs["tokenized_prompt"]
            recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return recombined

    def output_transform(self, outputs: dict) -> dict:
        """Apply the openpi output pipeline per-sample then recombine."""
        self._ensure_wrappers()
        batch_size = outputs["actions"].shape[0]
        transformed = []
        for i in range(batch_size):
            sample = tree_map(
                lambda x: _to_numpy(x[i]) if torch.is_tensor(x) else x[i],
                outputs,
            )
            sample = self._output_transform_fn(sample)
            transformed.append(sample)
        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if self.action_chunk is not None:
            recombined["actions"] = recombined["actions"][:, : self.action_chunk]
        return recombined

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict (from :meth:`input_transform`) into a device-resident :class:`Observation`."""
        device = self.device
        obs = Observation.from_dict(processed)

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        def _move_state(x):
            # The openpi Normalize stage runs in float64 (its norm_stats arrays
            # are float64); cast state back to float32 to match the legacy eval
            # processor (which did a final ``.float()``) and the model's compute
            # dtype. For pi05 the continuous state is unused (only the discrete
            # state tokens in the prompt are), but pi0 feeds it through
            # ``state_proj`` so float32 keeps the linear layer's dtype aligned.
            return (
                x.to(device=device, dtype=torch.float32)
                if isinstance(x, torch.Tensor)
                else x
            )

        return Observation(
            images={k: _move(v) for k, v in obs.images.items()},
            image_masks={k: _move(v) for k, v in obs.image_masks.items()},
            state=_move_state(obs.state),
            tokenized_prompt=_move(obs.tokenized_prompt),
            tokenized_prompt_mask=_move(obs.tokenized_prompt_mask),
            token_ar_mask=_move(obs.token_ar_mask),
            token_loss_mask=_move(obs.token_loss_mask),
            pcd_xyz=_move(obs.pcd_xyz),
        )

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions via the deterministic Euler ODE sampler.

        Only ``mode="eval"`` is supported at the eval level — that path is
        shared by every transforms-pipeline variant (eval task + RL eval
        fallback). ``mode="train"`` is implemented in the RL subclass which
        overrides this method to add the chain-collecting SDE sampler.
        Calling with ``mode="train"`` on the eval class raises
        :class:`NotImplementedError` so an eval-only model loudly refuses to
        be used for on-policy rollouts.
        """
        profile = self._latency_profile
        measure_latency = profile or self._latency_logger.enabled
        if measure_latency:
            torch.cuda.synchronize()
            total_started = time.perf_counter()
        del compute_values, kwargs  # accepted for call-site parity; eval ignores them
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use the RL subclass (actor.model.openpi.task='rl') for train rollouts."
            )
        # openpi.transforms pipeline (eval / RL).
        rtc_context = RTCModelContext.from_tensor(env_obs.get("rtc_context"))
        repacked = self._repack_env_obs(env_obs)
        if measure_latency:
            repack_finished = time.perf_counter()
        processed = self.input_transform(repacked, transpose=False)
        if measure_latency:
            input_transform_finished = time.perf_counter()
        observation = self._observation_dict_to_device(processed)
        if measure_latency:
            torch.cuda.synchronize()
            preprocess_finished = time.perf_counter()

        latency_metrics: dict[str, float] | None = {} if measure_latency else None
        actions, result = self._predict_eval(
            observation,
            noise=noise,
            rng=rng,
            rtc_context=rtc_context,
            latency_metrics=latency_metrics,
        )
        if measure_latency:
            torch.cuda.synchronize()
            total_finished = time.perf_counter()
            assert latency_metrics is not None
            latency_metrics.update({
                "repack": (repack_finished - total_started) * 1000.0,
                "input_transform": (input_transform_finished - repack_finished)
                * 1000.0,
                "to_device": (preprocess_finished - input_transform_finished) * 1000.0,
                "preprocess_total": (preprocess_finished - total_started) * 1000.0,
                "total": (total_finished - total_started) * 1000.0,
            })
            self._latency_logger.record(
                latency_metrics,
                metadata={
                    "batch_size": int(actions.shape[0]),
                    "num_steps": int(self.num_steps),
                    "action_shape": list(actions.shape),
                    "rtc_enabled": bool(
                        rtc_context is not None and rtc_context.enabled
                    ),
                },
            )
        if profile:
            assert latency_metrics is not None
            print(
                "[PI05 latency] "
                f"repack={latency_metrics['repack']:.2f} ms, "
                f"input_transform={latency_metrics['input_transform']:.2f} ms, "
                f"to_device={latency_metrics['to_device']:.2f} ms, "
                f"preprocess={latency_metrics['preprocess_total']:.2f} ms, "
                f"sample_actions={latency_metrics['sample_actions']:.2f} ms, "
                f"output_transform={latency_metrics['output_transform']:.2f} ms, "
                f"total={latency_metrics['total']:.2f} ms"
            )
        return actions, result

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        rtc_context: RTCModelContext | None = None,
        latency_metrics: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval and the RL eval path.

        Returns the env-frame ``actions`` plus the rollout-side return dict
        with ``prev_logprobs=None`` / ``prev_values=None`` and a minimal
        ``forward_inputs`` (action + raw model_action) — exactly what
        :class:`huggingface_worker.HuggingFaceWorker.predict` expects from an
        eval call.
        """
        measure_latency = latency_metrics is not None
        if measure_latency:
            model_started = time.perf_counter()

        if rtc_context is not None and rtc_context.enabled:
            if self._rtc_previous_model_actions is None:
                raise RuntimeError("RTC guidance requested before a first policy chunk")
            if self._rtc_previous_generation_id != rtc_context.previous_generation_id:
                raise RuntimeError(
                    "RTC previous generation mismatch: "
                    f"model={self._rtc_previous_generation_id}, "
                    f"request={rtc_context.previous_generation_id}"
                )
            previous_aligned = align_previous_chunk(
                self._rtc_previous_model_actions,
                rtc_context.execution_horizon_steps,
            )
            model_actions = self.model.sample_actions_rtc(
                observation,
                previous_aligned=previous_aligned,
                inference_delay_steps=rtc_context.predicted_delay_steps,
                prefix_attention_horizon=(
                    self.model.action_horizon - rtc_context.execution_horizon_steps
                ),
                prefix_attention_schedule=rtc_context.prefix_attention_schedule,
                max_guidance_weight=rtc_context.max_guidance_weight,
                num_steps=self.num_steps,
                noise=noise,
                rng=rng,
            )
        else:
            model_actions = self.model.sample_actions(
                observation, num_steps=self.num_steps, noise=noise, rng=rng
            )
        if rtc_context is not None:
            self._rtc_previous_model_actions = model_actions.detach().clone()
            self._rtc_previous_generation_id = rtc_context.response_generation_id
        if measure_latency:
            torch.cuda.synchronize()
            model_finished = time.perf_counter()
        env_outputs = self.output_transform({
            "actions": model_actions,
            "state": observation.state,
        })
        # openpi Unnormalize runs in float64; cast env actions back to float32 to
        # match the legacy eval processor's ``.astype(np.float32)`` contract (and
        # the action dtype the env/rollout worker expects).
        actions = env_outputs["actions"].to(device=self.device, dtype=torch.float32)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }
        if measure_latency:
            torch.cuda.synchronize()
            output_finished = time.perf_counter()
            assert latency_metrics is not None
            latency_metrics.update({
                "sample_actions": (model_finished - model_started) * 1000.0,
                "output_transform": (output_finished - model_finished) * 1000.0,
            })

        return actions, result

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head."""
        self._require_rlt()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)

        prepared_observation = pi0_model_module.preprocess_observation(
            observation, train=False
        )
        prefix_output, prefix_mask, kv_cache = self.model.build_prefix_cache(
            prepared_observation
        )
        rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, prepared_observation.tokenized_prompt
        )
        z_rl = self._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
            dtype=torch.float32
        )

        model_actions = self._sample_actions_from_prefix_cache(
            prepared_observation,
            prefix_mask,
            kv_cache,
        )
        ref_chunk = self.output_transform({
            "actions": model_actions,
            "state": observation.state,
        })["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }

    def _sample_actions_from_prefix_cache(
        self,
        observation: Observation,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run Pi0's eval Euler sampler using an already-built prefix cache."""
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.model.action_horizon,
                self.model.action_dim,
                device=device,
                generator=rng,
            )

        x_t = noise
        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out = self.model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.model.velocity_from_suffix(suffix_out)
            x_t = x_t + dt * v_t
            t += dt
        return x_t
