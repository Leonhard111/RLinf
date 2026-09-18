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

"""Collect a read-only AgileX Nero hardware/software baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

HOME = Path.home()
DEFAULT_OUTPUT = HOME / ".local" / "share" / "rlinf-nero" / "baseline"
REPOS = {
    "rlinf": HOME / "fjh" / "workbench" / "RLinf",
    "pi05": HOME / "pi05",
    "robotwin": HOME / "RoboTwin",
    "nero_aloha": HOME / "nero_aloha",
}
CRITICAL_FILES = {
    "vlastop_robot_config": HOME / "pi05" / "vlastop" / "config" / "robot.json",
    "pi05_norm_stats": HOME
    / "pi05"
    / "artifacts"
    / "checkpoints"
    / "nero_pi05"
    / "30000"
    / "assets"
    / "local"
    / "nero_aloha16"
    / "norm_stats.json",
    "pi05_checkpoint_metadata": HOME
    / "pi05"
    / "artifacts"
    / "checkpoints"
    / "nero_pi05"
    / "30000"
    / "params"
    / "_METADATA",
    "nero_launch": HOME
    / "nero_aloha"
    / "src"
    / "agx_arm_ros"
    / "src"
    / "agx_arm_ctrl"
    / "launch"
    / "start_nero_aloha.launch.py",
}
CHECKPOINT = HOME / "pi05" / "artifacts" / "checkpoints" / "nero_pi05" / "30000"
TOPICS = (
    "/nero_left/puppet/joint_states",
    "/nero_right/puppet/joint_states",
    "/nero_left/master/joint_states",
    "/nero_right/master/joint_states",
    "/nero_left/control/joint_states",
    "/nero_right/control/joint_states",
    "/looklook/risk_state",
)


def run(command: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    """Run a read-only command and retain output even when unavailable."""

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "error": str(error)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(path: Path, hash_contents: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    combined = hashlib.sha256()
    total_bytes = 0
    entries = []
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total_bytes += size
        digest = sha256_file(item) if hash_contents else None
        combined.update(relative.encode())
        combined.update(str(size).encode())
        if digest is not None:
            combined.update(digest.encode())
        entries.append({"path": relative, "size_bytes": size, "sha256": digest})
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": combined.hexdigest(),
        "content_hashed": hash_contents,
        "files": entries,
    }


def git_manifest(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "git": False, "tree": tree_manifest(path, False)}
    fields = {
        "commit": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(path), "branch", "--show-current"],
        "status": ["git", "-C", str(path), "status", "--short"],
        "remotes": ["git", "-C", str(path), "remote", "-v"],
        "submodules": ["git", "-C", str(path), "submodule", "status", "--recursive"],
    }
    return {
        "path": str(path),
        "git": True,
        **{name: run(command, 30) for name, command in fields.items()},
    }


def tcp_probe(host: str, port: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return {
                "reachable": True,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
            }
    except OSError as error:
        return {"reachable": False, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-checkpoint-content-hash", action="store_true")
    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%dT%H%M%S%z")
    output = args.output_dir / timestamp
    output.mkdir(parents=True, exist_ok=False)

    ros_topics = {}
    for topic in TOPICS:
        command = (
            "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
            "source /home/agilex/nero_aloha/install/setup.bash >/dev/null 2>&1; "
            f"timeout 3 ros2 topic info --verbose {topic}"
        )
        ros_topics[topic] = run(["bash", "-lc", command], 6)

    critical = {}
    for name, path in CRITICAL_FILES.items():
        critical[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    manifest = {
        "schema_version": "nero-baseline/1.0.0",
        "created_at": timestamp,
        "read_only_collection": True,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "repositories": {name: git_manifest(path) for name, path in REPOS.items()},
        "critical_files": critical,
        "checkpoint": tree_manifest(CHECKPOINT, not args.skip_checkpoint_content_hash),
        "hardware": {
            "gpu": run([
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]),
            "can": run(["ip", "-details", "link", "show", "type", "can"]),
            "usb": run(["lsusb"]),
        },
        "runtime": {
            "docker": run(["docker", "version", "--format", "{{json .}}"]),
            "docker_image": run([
                "docker",
                "image",
                "inspect",
                "rlinf/rlinf:agentic-rlinf0.4-maniskill_libero",
            ]),
            "conda": run([shutil.which("conda") or "conda", "env", "list"]),
            "ros_distro": os.environ.get(
                "ROS_DISTRO", "humble (expected; sourced per probe)"
            ),
        },
        "connectivity": {
            "ray_head_192.168.50.102_6379": tcp_probe("192.168.50.102", 6379),
            "lookcameras_127.0.0.1_8765": tcp_probe("127.0.0.1", 8765),
        },
        "ros_topics": ros_topics,
        "safety_responsibility": {
            "emergency_stop_owner": "operator (must be confirmed before any publish-mode test)",
            "risk_gate": "/looklook/risk_state; stale/invalid data must block publishing",
            "manual_reset": "operator acknowledgement required after STOP/FAULT",
            "phase_0_2_publishers_created": False,
        },
    }
    target = output / "baseline_manifest.json"
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    latest = args.output_dir / "LATEST"
    latest.write_text(str(output) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
