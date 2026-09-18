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

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo
from rlinf.utils.latency_logger import get_latency_logger


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key

        def optional_ordered_keys(name):
            value = cfg.get(name, None)
            if value is None:
                return None
            if isinstance(value, str):
                raise TypeError(f"{name} must be a list of observation keys")
            keys = tuple(str(key) for key in value)
            if not keys:
                raise ValueError(f"{name} must not be empty when configured")
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} contains duplicate keys: {keys}")
            return keys

        # Defaults stay None so every existing real-world task preserves the
        # previous sorted-state / extra-view behavior. Nero opts in explicitly.
        self.model_state_keys = optional_ordered_keys("model_state_keys")
        self.wrist_image_keys = optional_ordered_keys("wrist_image_keys")
        if self.wrist_image_keys and self.main_image_key in self.wrist_image_keys:
            raise ValueError("main_image_key must not also be a wrist_image_key")
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        """
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    def get_hold_actions(
        self, fallback_actions: np.ndarray | None = None
    ) -> np.ndarray:
        """Return per-env hold actions for smooth-intervene dummy chunks.

        Prefers an intervention wrapper's ``get_hold_action`` (e.g. absolute TCP
        hold). When unavailable, returns zeros so relative-action teleop setups
        keep a no-op command (same as the previous dummy-chunk behavior).
        """
        action_dim = int(self.action_space.shape[-1])
        holds: list[np.ndarray] = []
        for env_id, env in enumerate(self.env.envs):
            fallback = None
            if fallback_actions is not None:
                fallback = np.asarray(fallback_actions[env_id], dtype=np.float32)

            get_hold_action = None
            try:
                get_hold_action = env.get_wrapper_attr("get_hold_action")
            except AttributeError:
                get_hold_action = None

            if callable(get_hold_action):
                hold = np.asarray(get_hold_action(fallback), dtype=np.float32).reshape(
                    -1
                )
            else:
                hold = np.zeros(action_dim, dtype=np.float32)

            if hold.size != action_dim:
                raise ValueError(
                    "get_hold_actions expected action dim "
                    f"{action_dim}, got {hold.size} for env_id={env_id}."
                )
            holds.append(hold)
        return np.stack(holds, axis=0)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        state = raw_obs["state"]
        state_keys = self.model_state_keys or tuple(sorted(state))
        missing_state_keys = [key for key in state_keys if key not in state]
        if missing_state_keys:
            raise KeyError(
                f"model_state_keys missing from observation: {missing_state_keys}"
            )
        full_states = np.concatenate([state[key] for key in state_keys], axis=-1)
        obs["states"] = full_states

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]

        if self.wrist_image_keys is not None:
            missing_wrist_keys = [
                key for key in self.wrist_image_keys if key not in frames
            ]
            if missing_wrist_keys:
                raise KeyError(
                    f"wrist_image_keys missing from observation: {missing_wrist_keys}"
                )
            obs["wrist_images"] = np.stack(
                [frames[key] for key in self.wrist_image_keys], axis=1
            )
            excluded = {self.main_image_key, *self.wrist_image_keys}
            extra_images = [
                frames[key] for key in sorted(frames) if key not in excluded
            ]
            if extra_images:
                obs["extra_view_images"] = np.stack(extra_images, axis=1)
        else:
            raw_images = OrderedDict(sorted(frames.items()))
            raw_images.pop(self.main_image_key)
            if raw_images:
                obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        if "rtc" in raw_obs:
            obs["rtc_context"] = raw_obs["rtc"]

        # These arrays are snapshot-owned and remain alive until the transport
        # finishes. from_numpy avoids another full image copy on the AgileX CPU.
        obs = {
            key: torch.from_numpy(value)
            if isinstance(value, np.ndarray)
            else to_tensor(value)
            for key, value in obs.items()
        }
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        # max_episode_steps: null → external wrapper owns episode end.
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)
        if "rlt_switch_flags" in infos:
            infos["rlt_switch_flags"] = to_tensor(
                np.asarray(infos["rlt_switch_flags"], dtype=bool)
            )

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _notify_action_chunk_begin(self) -> None:
        """Tell intervention wrappers a new action chunk is starting."""
        for env in self.env.envs:
            try:
                on_begin = env.get_wrapper_attr("on_action_chunk_begin")
            except AttributeError:
                continue
            if callable(on_begin):
                on_begin()

    def _submit_native_action_chunk(self, chunk_actions) -> bool:
        """Let a single hardware env consume its native chunk immediately.

        Environments without ``submit_action_chunk`` keep the generic step loop.
        Nero exposes this hook because its host controller already consumes one
        complete policy generation, so waiting for the 50th Python/Gym step only
        adds latency and does not change the command sent to the robot.
        """

        if self.num_envs != 1:
            return False
        env = self.env.envs[0]
        try:
            submit_chunk = env.get_wrapper_attr("submit_action_chunk")
        except AttributeError:
            return False
        if not callable(submit_chunk):
            return False

        actions = chunk_actions
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(
                "native action chunk submission expects shape "
                "[1, chunk_size, action_dim]"
            )
        submit_chunk(actions[0])
        return True

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_started = time.perf_counter()
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_rlt_switch_flags = []
        self._notify_action_chunk_begin()
        notified = time.perf_counter()
        native_submitted = self._submit_native_action_chunk(chunk_actions)
        submitted = time.perf_counter()
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        expanded = time.perf_counter()
        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
            infos_list[-1] = infos_last
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = torch.stack(
                raw_chunk_rlt_switch_flags, dim=1
            )
            infos_list[-1] = infos_last

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        if native_submitted:
            finished = time.perf_counter()
            get_latency_logger("realworld_chunk").record(
                {
                    "notify_chunk_begin": (notified - chunk_started) * 1000.0,
                    "native_submit": (submitted - notified) * 1000.0,
                    "expand_gym_steps": (expanded - submitted) * 1000.0,
                    "stack_and_finalize": (finished - expanded) * 1000.0,
                    "chunk_step_total": (finished - chunk_started) * 1000.0,
                },
                metadata={
                    "chunk_size": int(chunk_size),
                    "action_shape": list(chunk_actions.shape),
                },
            )
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
