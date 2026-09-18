# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""OpenPI data configuration for AgileX Nero (7+gripper per arm)."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import nero_policy


@dataclasses.dataclass(frozen=True)
class LeRobotNeroDataConfig(DataConfigFactory):
    """Nero transforms with a 16D physical space and a 32D model space."""

    default_prompt: str | None = None
    extra_delta_transform: bool = True
    adapt_to_pi: bool = False
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if self.adapt_to_pi:
            raise ValueError("pi05_nero requires adapt_to_pi=False")
        data_transforms = _transforms.Group(
            inputs=[nero_policy.NeroInputs(adapt_to_pi=False)],
            outputs=[nero_policy.NeroOutputs(adapt_to_pi=False)],
        )
        if self.extra_delta_transform:
            delta_mask = np.asarray(
                [True] * 7 + [False] + [True] * 7 + [False], dtype=bool
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_mask)],
                outputs=[_transforms.AbsoluteActions(delta_mask)],
            )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
