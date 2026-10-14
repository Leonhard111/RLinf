# Nero pi0.5 inference latency experiment

This experiment is confined to the single RTX 4090 (48 GB) RLinf checkout. It
uses the existing `nero_right_rlt_stage1_20000` checkpoint and `norm_stats.json`.
It does not alter trained weights, normalization, action space, diffusion step
count, or the native bridge. The existing live runner's new
`--compile-suffix` and `--compile-prefix` switches are **off by default**.

## Finding

The 5-step sampler consists of one visual/text prefix and five repeated action
expert suffix passes. On the single 4090, the unchanged bf16 checkpoint with a
224x224 two-camera synthetic observation measured:

| Offline measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Original 5-step fixed-input benchmark | 206.0 ms | 210.3 ms | 210.5 ms |
| Compiled suffix, 5-step random-image benchmark | 67.6 ms | 68.2 ms | 69.6 ms |
| Actual offline runner, native, 29 warm calls | 210.4 ms | 212.5 ms | 213.3 ms |
| Actual runner, suffix only, 1000 calls | 66.7 ms | 67.3 ms | **115.8 ms** |
| Random-image benchmark, prefix + suffix, 1000 calls | 44.2 ms | 44.9 ms | 46.3 ms |
| Actual runner, prefix + suffix, 3000 calls | **44.5 ms** | **45.1 ms** | **94.0 ms** |
| Full single-host path, prefix + suffix, 3000 calls | **45.2 ms** | **45.8 ms** | **92.9 ms** |

`torch.compile(mode="reduce-overhead")` of the repeated action-expert suffix
reduced steady latency from about 210 ms to 67 ms. Its visual prefix still took
about 53 ms. Compiling that prefix as well reduced it to about 32 ms, while
five compiled suffix passes together took about 11 ms. The prefix path uses
`torch.compiler.cudagraph_mark_step_begin()` and clones returned tensors
before they enter the suffix so CUDA Graph storage cannot be overwritten while
still in use. This follows PyTorch's [CUDA Graph Trees guidance](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html).

Compilation is a one-time startup cost: about 29 seconds for suffix only and
about 70 seconds for prefix + suffix on this host. The first prefix + suffix
model call after warmup took 88.4 ms and is included in the 3000-call result.
The original uncompiled runner's first request took 9.0 seconds due to its
existing kernels compiling on demand; its baseline row excludes that cold call.

The suffix-only 1000-call runner test recorded one call over 100 ms (115.8 ms;
P99 was 68.3 ms). Therefore suffix-only compilation did **not** meet the
requested observed maximum. The prefix + suffix runner's 3000 calls reached
94.0 ms at worst, with no calls over 100 ms. Its full single-host path,
including observation adaptation and absolute-action conversion, reached
92.9 ms at worst over another 3000 calls. These are measured samples, not a
hard real-time guarantee.
An additional 1000-call offline profiler recorded wall, thread CPU, and CUDA
Event times: median wall 73.4 ms, P99 76.1 ms, maximum 79.6 ms. Its per-call
CUDA Event instrumentation adds overhead, so these numbers are not directly
comparable with the ordinary runner. It did not reproduce the isolated slow
call or establish its cause. Raw measurements are in `results/`.

A third, 1000-call offline run timed the complete single-host path from
`right_model_observation` through prediction and `right_actions_to_native`:
median 67.9 ms, P99 69.9 ms, maximum 76.0 ms. The input adapter took a median
0.68 ms and action conversion 0.08 ms; the model remained the dominant cost.
This used synthetic 224x224 images and did not include camera capture, JPEG
transport, bridge processing, or execution of the action chunk. It also did
not reproduce the earlier 115.8 ms outlier.

The longer 3000-call suffix-only full-path run **did** reproduce one outlier: median
68.25 ms, P99 69.40 ms, maximum 117.59 ms at call 910. That call spent
116.54 ms inside the model, with normal action conversion and no Python GC
collection. The 115.8 ms excursion is therefore not explained by the simple
observation adapter, native-action conversion, or Python garbage collection.
The cause inside the model/GPU path is still under investigation.

Across 20 varied image, joint-state, and noise inputs, the largest
elementwise action difference versus native five-step inference was 0.0062
for suffix only and 0.0107 for prefix + suffix (radians for joint actions,
gripper units for gripper). Average absolute difference across the 20
prefix + suffix cases was 0.00078. Outputs changed across every input. This
is a numerical parity check, not a robot-task success evaluation; the extra
prefix compilation remains opt-in until task behavior can be assessed.

We also decoded 14 paired front/wrist frames from the saved
`nero_right_smoke_video` episode on this host. With the deployed red-on-green
prompt and inference-mode noise, compiled model calls ranged from 43.3 to
44.3 ms across those changing frames; the largest joint-action difference
was 0.0093 rad. The saved episode's own prompt includes a second stacking
step, so this is image-distribution coverage, not a task success check.
An earlier version of that diagnostic created fixed noise *outside*
`torch.inference_mode()` and caused a 3.87-second suffix recompilation.
`TORCH_LOGS=recompiles` identified an `x_t` dispatch-key mismatch. Moving
diagnostic noise creation inside inference mode removed the recompile;
the live runner already samples noise in inference mode.

PyTorch SDPA replacement did not improve this checkpoint: 207.2 ms median
for five steps and a slower individual suffix. A one-step native sampler took
85.3 ms but changed actions appreciably, so the deployment path retains five
steps. No quantization or pruning is required for the observed offline model
latency reduction; those approaches would require separate quality testing.

The isolated `phase_profile.py` confirmed that the visual prefix dominates
steady latency and that the five suffix passes together are about 11 ms.

## RLT Stage 2 relevance

The frozen Stage 1 model used by RLinf's RLT Stage 2 rollout calls
`extract_rlt_obs`, which uses the same visual prefix and action suffix.
An offline probe with this Nero checkpoint measured 50 calls: native median
250.3 ms versus compiled prefix + suffix median 47.5 ms. The reference action
chunk differed by at most 0.0053, but the `z_rl` feature changed more: mean
absolute difference 0.0267, maximum 0.1484. On 14 saved camera frames, the
median relative L2 difference in `z_rl` was 4.3% (maximum 5.6%).

A separate 50-call suffix-only probe measured 70.8 ms median and reproduced
native `z_rl` exactly; its reference action chunk differed by at most 0.0028.
This is the safer candidate for a policy already trained on native features.
These probes use the Stage 1 feature method with five denoising steps, not a
complete RLT Stage 2 training run or its potentially different configuration.
Neither optimization is enabled in Stage 2 workers yet.

## Run offline

From `/home/rcir/Data_Disk/fjh/workbench/RLinf`:

```bash
bash run/nero/run_right_pi05_container.sh \
  --mode offline --max-generations 30 \
  --compile-suffix --compile-prefix
```

The line `compiled_prefix_suffix_warmup_s=...` is a one-time startup cost.
The `inference_ms=...` lines are synchronized model-call wall latency.
For the previously tested 67 ms path, pass `--compile-suffix` alone. Omitting
both compile flags restores the original inference path. `--compile-prefix`
requires `--compile-suffix`.

### Reuse compiled kernels after a container restart

The launcher mounts a writable `torch_compile_cache/` subdirectory of the
selected checkpoint whenever either compile flag is present. PyTorch Inductor
and Triton write generated kernels there; the checkpoint weights and norm file
stay on a separate read-only mount. The same directory also holds a small
`runtime_home/` for tokenizer and other per-user caches. On an empty cache, the
launcher copies the bundled PaliGemma tokenizer from the Docker image into
`runtime_home/`; it does not need to download this asset from GCS. Compile-enabled
containers run as the host user so these files remain user-owned. First populate
the cache without connecting to a robot:

```bash
bash run/nero/run_right_pi05_container.sh \
  --mode offline --max-generations 1 \
  --compile-suffix --compile-prefix
```

Repeat the same command to see the cached startup time in
`compiled_prefix_suffix_warmup_s`. Later inference invocations use the same
flags and checkpoint path to reuse the cache. The optional
`NERO_RIGHT_COMPILE_CACHE_DIR` environment variable moves the cache to another
host directory. Without compile flags, the launcher does not create or mount
the cache.

On the single 4090, a fresh cache took 71.8 seconds to compile and warm, while
a second disposable container using the same cache took 12.2 seconds to warm.
Both produced a 50-step chunk in roughly 45 ms after warmup. These timings
exclude model loading and container startup. The cache held approximately
34 MB of generated artifacts after the first run.

This is a compiler cache, not a second checkpoint or a serialized model. Each
new process still loads weights, traces/checks the compiled callable, and
warms CUDA Graphs; a cache hit therefore reduces startup work but does not
remove startup time. Changes to model code, input shapes, PyTorch/Triton, or
GPU may require fresh compilation. The cache can be removed independently of
`full_weights.pt` and `norm_stats.json`.

To repeat the fixed-input benchmark with the same checkpoint and normalization:

```bash
docker run --rm --gpus all --network none --shm-size=4g \
  --mount "type=bind,src=$PWD,dst=/workspace/RLinf" \
  --mount "type=bind,src=/home/rcir/Data_Disk/fjh/checkpoints/nero_right_rlt_stage1_20000,dst=/workspace/checkpoint,readonly" \
  --workdir /workspace/RLinf --entrypoint bash \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero \
  -lc 'source switch_env openpi && export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-} && python experiments/pi05_latency/benchmark.py --compile-suffix --compile-prefix --prefix-cudagraph-safe --validate-dynamic --dynamic-cases 20 --random-observation --steps 5 --warmup 6 --repeats 100 --output experiments/pi05_latency/results/compiled.json'
```

`tail_profile.py`, `phase_profile.py`, `pipeline_profile.py`,
`stage2_feature_probe.py`, and `real_frame_parity.py` provide offline
diagnostics with results under `results/`. None opens a robot bridge
connection.

## Use with the existing bridge

When a separate, appropriately configured AgileX bridge is available and the
robot is free, append `--compile-suffix --compile-prefix` to the existing
`run_right_pi05_container.sh` invocation.
For example, a **shadow** session can measure real image/network effects
without publishing commands:

On AgileX, launch the native bridge with its usual ROS environment and prompt,
but omit `--publish` (that selects SHADOW mode):

```bash
cd ~/fjh/workbench/pi05/bridge
bash run_native_nero_bridge.sh \
  --prompt 'Stack the red square on top of the green square.' \
  --bind tcp://0.0.0.0:5555 \
  --action-timeout 180 \
  --sync-interpolation pchip \
  --toppra-execution-steps 50 --monitor
```

On the single 4090, from the RLinf checkout:

```bash
bash run/nero/run_right_pi05_container.sh \
  --mode shadow --max-generations 30 \
  --bridge-request-timeout-s 180 \
  --compile-suffix --compile-prefix
```

The bridge must itself be running in SHADOW mode; the client checks the mode
and refuses a mismatch. This experiment has **not** yet measured the real
bridge/network tail latency or task success. The prefix + suffix variant's
observed offline maximum was under 100 ms, but no strict maximum is proven
for the real input and complete robot loop. If actual `inference_ms` exceeds
100 ms, record its distribution before changing the number of denoising steps.

The checkpoint and norm defaults live in `run/nero/run_right_pi05_container.sh`.
If deploying another checkpoint, set `NERO_RIGHT_CHECKPOINT_DIR` to the
directory containing both `full_weights.pt` and `norm_stats.json`. The
experiment does not enable compilation for RL Stage 2 training workers;
their RL/sampler path requires its own correctness and memory check.
