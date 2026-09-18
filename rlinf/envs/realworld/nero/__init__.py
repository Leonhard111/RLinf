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

"""AgileX Nero data and action contracts.

This package intentionally has no ROS2, camera SDK, or CAN dependency.  The
host bridge and the RLinf worker must both validate data through this module.
"""

from .contract import (
    ACTION_HORIZON,
    CAMERA_NAMES,
    CONTRACT,
    MODEL_ACTION_DIM,
    PHYSICAL_ACTION_DIM,
    PROTOCOL_VERSION,
    ControlMode,
    NeroActionChunk,
    NeroContract,
    NeroErrorCode,
    NeroObservation,
    NeroRiskState,
    NeroRTCContext,
    NeroStepRecord,
    absolute_to_delta,
    delta_to_absolute,
    pad_physical_action,
    unpad_model_action,
)
from .nero_client import NeroBridgeCapabilities, NeroBridgeError, NeroRobotClient

__all__ = [
    "ACTION_HORIZON",
    "CAMERA_NAMES",
    "CONTRACT",
    "MODEL_ACTION_DIM",
    "PHYSICAL_ACTION_DIM",
    "PROTOCOL_VERSION",
    "ControlMode",
    "NeroActionChunk",
    "NeroBridgeCapabilities",
    "NeroBridgeError",
    "NeroContract",
    "NeroErrorCode",
    "NeroObservation",
    "NeroRTCContext",
    "NeroRiskState",
    "NeroRobotClient",
    "NeroStepRecord",
    "absolute_to_delta",
    "delta_to_absolute",
    "pad_physical_action",
    "unpad_model_action",
]
