# AgileX Nero: phases 0–4

The production control worker uses the official RLinf 0.4 Docker image.  The
repository is mounted read-only and no CAN or USB device is mapped into the
container.  ROS2, LookCameras, and future control publishing stay in a separate
host bridge.

## Environment

The domestic mirror is used for the one-time pull, then tagged with the stable
upstream name:

```bash
docker pull docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
docker tag docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

# The upstream image lacks pyzmq.  Build the thin, pinned control image.
docker build \
  -f run/nero/Dockerfile.control \
  -t rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control \
  .
./run/nero/preflight.sh
./run/nero/run_control_container.sh
```

The official image contains several environments.  All RLinf commands here
first run `source switch_env openpi`, selecting the PyTorch/OpenPI environment
used by RLinf RLT.  Conda environments already present on the robot are not
modified and are not part of the production path.

## Baseline

Static baseline collection is read-only and may be run with all robot services
stopped:

```bash
python3 run/nero/collect_baseline.py
```

Checkpoint content hashing is enabled by default.  It can take several minutes.
Use `--skip-checkpoint-content-hash` only for a quick diagnostic manifest.

The 100-frame collector requires an operator to have already started the
existing ROS2 state nodes and LookCameras service.  It creates subscriptions
only and exits if either control topic has a publisher:

```bash
source /opt/ros/humble/setup.bash
source ~/nero_aloha/install/setup.bash
conda activate <existing-vlastop-environment>
python3 run/nero/collect_observations.py --count 100 --prompt '<task prompt>'
```

After the existing π0.5 service has been started by the operator, reproduce its
actions on those fixed samples without touching ROS2:

```bash
python3 run/nero/run_baseline_pi05.py \
  ~/.local/share/rlinf-nero/baseline_observations/<timestamp> \
  --output ~/.local/share/rlinf-nero/baseline_pi05_actions.npz \
  --log ~/.local/share/rlinf-nero/baseline_pi05_dry_run.json
```

## Contract tests

```bash
docker run --rm --network host --shm-size 20g \
  --mount type=bind,src="$HOME/fjh/workbench/RLinf",dst=/workspace/RLinf,readonly \
  --workdir /workspace/RLinf --entrypoint bash \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero \
  -lc 'source switch_env openpi && PYTHONPATH=/workspace/RLinf pytest -q tests/unit_tests/test_nero_contract.py'
```

Nothing in phases 0–2 creates a ROS2 publisher or sends CAN commands.

## Phase 3: VLASTOP bridge and NeroEnv

Phase 3 adds a transport adapter; it does not replace the existing controller.
The host bridge injects `RemotePolicyAdapter` through VLASTOP's existing
`policy_factory` and extends `NeroPi05Node` only with read-only master-state
subscriptions. `ControlRuntime`, RoboTwin TOPP, the risk gate, stop planner,
publisher watchdog, and the 200 Hz ROS publisher remain the original files.

The host process deliberately does not import the RLinf training package. It
loads only the pure-NumPy Nero contract by file path and uses standard TCP plus
framed msgpack, so the existing `RoboTwin` conda environment is unchanged.

Software-only acceptance:

```bash
docker run --rm --network host \
  --mount type=bind,src="$HOME/fjh/workbench/RLinf",dst=/workspace/RLinf,readonly \
  --workdir /workspace/RLinf --entrypoint bash \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control \
  -lc 'source switch_env openpi && PYTHONPATH=/workspace/RLinf \
    pytest -q tests/unit_tests/test_nero_contract.py \
      tests/unit_tests/test_nero_client_env.py'

docker run --rm --network host \
  --mount type=bind,src="$HOME/fjh/workbench/RLinf",dst=/workspace/RLinf,readonly \
  --workdir /workspace/RLinf --entrypoint bash \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control \
  -lc 'source switch_env openpi && PYTHONPATH=/workspace/RLinf \
    python run/nero/run_dummy_env.py --steps 10000'
```

For the live shadow acceptance, the operator first starts LookCameras,
LookLook, and the existing puppet/master ROS state nodes. Then start the host
bridge without `--publish`:

```bash
cd /home/agilex/pi05
./bridge/run_nero_bridge.sh --prompt 'task instruction'
```

The default endpoint is loopback-only `tcp://127.0.0.1:5555`; the RLinf control
container therefore uses host networking. In a second terminal run:

```bash
cd /home/agilex/fjh/workbench/RLinf
./run/nero/run_control_container.sh
# Inside the container:
python run/nero/run_shadow_hold.py --steps 10
```

After the first valid client request, a separate 20 Hz heartbeat guard requires
client activity within 0.75 seconds. `NeroRobotClient` runs an independent 10 Hz
status heartbeat, so a slow model inference does not look like a dead client. On
timeout the guard calls VLASTOP's existing `request_safety_stop()`; the RLinf
client does not publish or synthesize a replacement trajectory.

`run_shadow_hold.py` refuses to connect if the bridge reports publish mode. Do
not use the bridge's `--publish` option in phase 3. Real publisher and hardware
motion acceptance belongs to phase 4 and requires separate authorization.

## PyTorch π0.5 Nero adapter

`pi05_nero` keeps the model action space at 32 dimensions and maps the physical
Nero interface to 16 dimensions (`7 joints + gripper` per arm). Only
`state.puppet` enters the model. The image order is fixed to `cam_high`,
`cam_left_wrist`, `cam_right_wrist`; master and risk remain available to the
intervention/safety path but are not concatenated into π0.5 proprioception.

Hardware-free acceptance on 108:

```bash
docker run --rm --network host --shm-size 20g \
  --mount type=bind,src="$HOME/fjh/workbench/RLinf",dst=/workspace/RLinf,readonly \
  --workdir /workspace/RLinf --entrypoint bash \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero-nero-control \
  -lc 'source switch_env openpi && PYTHONPATH=/workspace/RLinf pytest -q \
    tests/unit_tests/test_nero_contract.py \
    tests/unit_tests/test_nero_client_env.py \
    tests/unit_tests/test_nero_openpi_transform.py \
    tests/unit_tests/test_nero_pi05_checkpoint.py'
```

Strict checkpoint loading and GPU timing run on 102, where the converted model
is stored:

```bash
python run/nero/verify_pi05_checkpoint.py \
  --model /workspace/fjh/checkpoints/pi05_base_openpi_rlinf \
  --norm-stats /workspace/fjh/checkpoints/pi05_base_openpi_rlinf/local/nero_aloha16/norm_stats.json \
  --device cuda --warmup 3 --runs 100 \
  --prompt 'put the object at the target position'
```
