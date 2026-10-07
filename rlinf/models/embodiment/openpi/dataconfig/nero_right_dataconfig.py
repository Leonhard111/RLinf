# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""OpenPI LeRobot data config for 2-camera, right-arm-only Nero RLT Stage 1."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies.nero_right_policy import (
    NeroRightInputs,
    NeroRightOutputs,
    NeroRightTrainingDropout,
)


@dataclasses.dataclass(frozen=True)
class LeRobotNeroRightDataConfig(DataConfigFactory):
    default_prompt: str | None = None
    wrist_dropout_prob: float = 0.0
    state_dropout_prob: float = 0.0
    use_delta_joint_actions: bool = True
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[NeroRightInputs()], outputs=[NeroRightOutputs()]
        )
        if self.use_delta_joint_actions:
            joint_mask = np.asarray([True] * 7 + [False], dtype=bool)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(joint_mask)],
                outputs=[_transforms.AbsoluteActions(joint_mask)],
            )
        standard = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        dropout = NeroRightTrainingDropout(
            wrist_probability=self.wrist_dropout_prob,
            state_probability=self.state_dropout_prob,
        )
        model_transforms = _transforms.Group(
            inputs=(dropout, *standard.inputs), outputs=standard.outputs
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
