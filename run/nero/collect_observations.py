#!/usr/bin/env python3
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

"""Collect synchronized Nero regression observations without any publisher.

Prerequisites are existing, operator-started LookCameras and ROS2 state nodes.
This program only creates subscriptions.  It refuses to run if either control
topic already has a publisher, which keeps baseline capture separate from any
motion-producing process.  A dedicated ROS executor keeps all four 200 Hz state
subscriptions fresh while the main thread samples cameras and writes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
VLASTOP_ROOT = HOME / "pi05" / "vlastop"
CAMERA_NAMES = ("cam_high", "cam_right_wrist", "cam_left_wrist")
JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
)
STATE_TOPICS = {
    "left_puppet": "/nero_left/puppet/joint_states",
    "right_puppet": "/nero_right/puppet/joint_states",
    "left_master": "/nero_left/master/joint_states",
    "right_master": "/nero_right/master/joint_states",
}
CONTROL_TOPICS = ("/nero_left/control/joint_states", "/nero_right/control/joint_states")


def ordered(message: object) -> np.ndarray:
    names = list(getattr(message, "name"))
    positions = np.asarray(getattr(message, "position"), dtype=np.float32)
    if len(names) != 8 or positions.shape != (8,) or set(names) != set(JOINT_NAMES):
        raise ValueError("JointState must contain exactly joint1..joint7,gripper")
    values = np.asarray(
        [positions[names.index(name)] for name in JOINT_NAMES], dtype=np.float32
    )
    if not np.isfinite(values).all():
        raise ValueError("JointState contains NaN/Inf")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HOME / ".local" / "share" / "rlinf-nero" / "baseline_observations",
    )
    parser.add_argument("--interval-s", type=float, default=0.1)
    args = parser.parse_args()
    if args.count < 1 or args.interval_s <= 0:
        parser.error("count and interval must be positive")

    sys.path.insert(0, str(VLASTOP_ROOT))
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from vlastop_robot.cameras import LookCamerasGroup
    from vlastop_robot.config import VlastopConfig

    config = VlastopConfig.load(VLASTOP_ROOT / "config" / "robot.json")
    fatal: list[BaseException] = []
    cameras = LookCamerasGroup(config.camera, fatal.append)
    rclpy.init()
    node = Node("nero_baseline_read_only_collector")
    rclpy.spin_once(node, timeout_sec=0.5)
    publisher_counts = {}
    for topic in CONTROL_TOPICS:
        publisher_counts[topic] = int(node.count_publishers(topic))
        if publisher_counts[topic] != 0:
            raise RuntimeError(f"refusing baseline capture: {topic} has a publisher")

    lock = threading.Lock()
    states: dict[str, tuple[np.ndarray, int]] = {}

    def callback(name: str):
        def accept(message: JointState) -> None:
            value = ordered(message)
            with lock:
                states[name] = (value, time.monotonic_ns())

        return accept

    subscriptions = [
        node.create_subscription(JointState, topic, callback(name), 10)
        for name, topic in STATE_TOPICS.items()
    ]
    del subscriptions

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_ros() -> None:
        try:
            executor.spin()
        except BaseException as error:
            fatal.append(error)

    spin_thread = threading.Thread(
        target=spin_ros,
        name="nero-state-executor",
        daemon=True,
    )
    spin_thread.start()

    try:
        cameras.start()
        cameras.wait_for_first_frames(config.camera.startup_timeout_s)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if fatal:
                raise RuntimeError(str(fatal[0]))
            with lock:
                if set(states) == set(STATE_TOPICS):
                    break
            time.sleep(0.01)
        else:
            raise TimeoutError(f"missing ROS states: {set(STATE_TOPICS) - set(states)}")

        timestamp = time.strftime("%Y%m%dT%H%M%S%z")
        output = args.output_dir / timestamp
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": "nero-observations/1.0.0",
            "prompt": args.prompt,
            "read_only_collection": True,
            "control_publisher_counts": publisher_counts,
            "samples": [],
        }
        for index in range(args.count):
            started = time.monotonic()
            if fatal:
                raise RuntimeError(str(fatal[0]))
            snapshot = cameras.snapshot()
            with lock:
                current = {
                    name: (value.copy(), stamp)
                    for name, (value, stamp) in states.items()
                }
            observation_ns = time.monotonic_ns()
            state_ages_ms = {
                name: (observation_ns - stamp) / 1e6
                for name, (_, stamp) in current.items()
            }
            if any(age < 0 or age > 250.0 for age in state_ages_ms.values()):
                raise RuntimeError(f"ROS state age out of range: {state_ages_ms}")
            puppet = np.concatenate((
                current["left_puppet"][0],
                current["right_puppet"][0],
            ))
            master = np.concatenate((
                current["left_master"][0],
                current["right_master"][0],
            ))
            state_stamps = np.asarray(
                [current[name][1] for name in STATE_TOPICS], dtype=np.int64
            )
            sample_path = output / f"sample_{index:06d}.npz"
            np.savez_compressed(
                sample_path,
                cam_high=snapshot.images[0],
                cam_right_wrist=snapshot.images[1],
                cam_left_wrist=snapshot.images[2],
                camera_timestamps_ns=np.asarray(snapshot.timestamps_ns, dtype=np.int64),
                puppet_state=puppet.astype(np.float32),
                master_state=master.astype(np.float32),
                state_timestamps_ns=state_stamps,
                observation_timestamp_ns=np.asarray(observation_ns, dtype=np.int64),
                prompt=np.asarray(args.prompt),
            )
            manifest["samples"].append({
                "path": sample_path.name,
                "sha256": sha256_file(sample_path),
                "camera_ages_ms": list(snapshot.ages_ms),
                "camera_skew_ms": snapshot.skew_ms,
                "state_ages_ms": state_ages_ms,
            })
            remaining = args.interval_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        manifest_path = output / "observations_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest_path)
    finally:
        cameras.close()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
