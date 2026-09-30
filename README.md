# EWAM

**EWAM** is a trimodal flow-matching Diffusion Transformer for robotic manipulation. A single denoising transformer processes three token streams jointly:

- **Video model** — [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2), denoises future video latents;
- **Action expert** — a lightweight transformer expert that denoises the action chunk;
- **VLM** — [Qwen3-VL-2B-Instruct](https://github.com/QwenLM/Qwen3-VL) (trainable), fused into every layer through direct MoT (per-layer QKV projections).

The three streams attend through an **asymmetric attention mask**: the Action stream attends to everything (video + VLM + action), while Video and VLM only self-attend. Text conditioning for the video model's cross-attention comes from pre-encoded umt5-xxl (T5) instruction embeddings.

Actions and states are unified into a padded **16-dim space** so that one model covers different robot embodiments; out-of-dataset dimensions are zero-padded and masked out of the loss. This repository ships **four end-to-end pipelines**:

| | Training | Evaluation |
|---|---|---|
| **RoboTwin2.0** | stage-2 finetune with reweighted loss — `scripts/train_ewam_robotwin.sh` | benchmark deployment — `inference/robotwin/` (`deploy_policy.py` + `auto_eval.sh`) |
| **LIBERO** | stage-2 finetune on lerobot-format demos — `scripts/train_ewam_libero.sh` | 4 suites × 10 tasks, dim16 horizontal variant — `scripts/run_ewam_libero_eval_horizontal_dim16.sh` |

**Model components** (parameter counts measured from the released stage-2 checkpoint):

| Component | Base model | Parameters |
|---|---|---|
| **Video model** | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | ~5.00B |
| **VLM** | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | ~2.44B |
| **VLM fusion** (per-layer direct-MoT QKV projections) | — | ~0.76B |
| **Action expert** | — | ~0.64B |
| **Total** | | **~8.8B** |

**Hardware reference**:

| Mode | VRAM | Recommended GPU |
|---|---|---|
| Evaluation (T5 encoded on the fly) | ~41 GB | A100-80G / H100 |
| Training | > 80 GB | A100-80G / H100 |

## Table of Contents

- [Installation](#installation)
- [RoboTwin](#robotwin)
- [LIBERO](#libero)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Repository layout

```
EWAM/
├── train/
│   ├── train_ewam.py                              # Training entry point (single entry for both datasets)
│   └── sample.py                                  # Validation / sampling utilities
├── eval_scripts/
│   ├── eval_ewam_libero_single_horizontal_dim16.py  # Single (suite, task) LIBERO eval (hydra)
│   └── attention_analysis_mask.py                 # Optional attention visualization for eval
├── scripts/
│   ├── train_ewam_robotwin.sh                     # Training launcher (RoboTwin, torchrun + DeepSpeed ZeRO-1)
│   ├── train_ewam_libero.sh                       # Training launcher (LIBERO)
│   └── run_ewam_libero_eval_horizontal_dim16.sh   # 4-suite parallel eval scheduler
├── inference/
│   └── robotwin/
│       ├── deploy_policy.py                       # RoboTwin benchmark policy (EWAM)
│       ├── deploy_policy.yml                      # RoboTwin eval config (template)
│       ├── auto_eval.sh                           # 50-task parallel eval launcher
│       ├── eval.sh                                # Single-task eval launcher
│       ├── paths_config.example.yml               # Copy to paths_config.yml and edit
│       ├── requirements.txt                       # RoboTwin-env inference deps
│       └── tasks_all.txt                          # 50 RoboTwin2.0 task names
├── configs/
│   ├── ewam_robotwin.yaml                         # Training config (RoboTwin dim16)
│   ├── ewam_libero.yaml                           # Training config (LIBERO dim16)
│   ├── ewam_libero_eval_horizontal_dim16.yaml     # LIBERO eval config (dim16 horizontal)
│   └── zero1.json                                 # DeepSpeed ZeRO stage-1 config
├── models/
│   ├── ewam.py                                    # EWAM model (asymmetric mask + inference KV-cache;
│                                                  #   loss_reweight selects the loss variant)
│   ├── wan_model_mask.py                          # WAN video-model wrapper
│   ├── action_expert.py                           # Action expert
│   └── qwen3_module_wan.py                        # Qwen3-VL per-layer fusion module
├── data/
│   ├── dataset.py                                 # Dataset factory (dispatch by dataset.type)
│   ├── robotwin2/
│   │   ├── robotwin_agilex_dataset_dim16.py       # RoboTwin dim16 training dataset
│   │   └── robotwin_data_convert/                 # RoboTwin2.0 → EWAM data conversion pipeline
│   ├── libero/
│   │   ├── libero_lerobot_dataset.py              # LIBERO lerobot-format training dataset
│   │   └── regenerate_libero_t5_with_prefix.py    # T5 embedding generation for LIBERO suites
│   └── utils/image_utils.py
├── experiments/libero/libero_utils.py             # LIBERO env / rollout helpers
├── utils/                                         # common / scheduler / vlm_utils
├── wan/                                           # Wan2.2 official code (Alibaba), vendored
├── requirements.txt
└── README.md
```

## Installation

### 1. Python environment

```bash
conda create -n ewam python=3.10 -y
conda activate ewam

# torch (CUDA 12.6 wheels — adjust the index-url for your CUDA version)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# optional: flash-attention kernels (Wan attention falls back to PyTorch SDPA without it)
pip install flash-attn --no-build-isolation

pip install -r requirements.txt
```

### 2. Pretrained weights

**Released EWAM checkpoints** (stage-1 pretrain + the three stage-2 finetunes) are hosted on HuggingFace: [HaoWang00/EWAM](https://huggingface.co/HaoWang00/EWAM)

```bash
pip install -U "huggingface_hub"
huggingface-cli download HaoWang00/EWAM --local-dir /path/to/ewam_weights
```

| Checkpoint | Usage |
|---|---|
| Pretrain (multi-source, stage-1) | Init for stage-2 finetune — `finetune.checkpoint_path` |
| RoboTwin c2r | RoboTwin 2.0 evaluation — `checkpoint_path` in `paths_config.yml` (see [RoboTwin](#robotwin) → Evaluation) |
| RoboTwin in-domain | RoboTwin 2.0 evaluation — `checkpoint_path` in `paths_config.yml` (see [RoboTwin](#robotwin) → Evaluation) |
| LIBERO | LIBERO evaluation — `CKPT` override of the eval launcher (see [LIBERO](#libero) → Evaluation) |

Each checkpoint is a DeepSpeed-style directory containing `mp_rank_00_model_states.pt` — point the corresponding path at that directory (or the `.pt` file itself).

Two init modes are supported for stage-2 training:

| Mode | Config field | What loads |
|---|---|---|
| **Stage-2 finetune** (default) | `finetune.checkpoint_path` → stage-1 pretrain dir (the table above) | WAN + VLM weights come from the finetune checkpoint, experts included |
| **From scratch** | `finetune.checkpoint_path: null` | WAN + VLM load official pretrained backbones; action expert / fusion modules are randomly initialized |

**Backbone weights** are not part of the EWAM repo — download them from their official sources and point the config fields at them (all paths in this repo's configs are `/path/to/...` placeholders — replace them with your own):

| Component | Config fields | Notes |
|---|---|---|
| Wan2.2-TI2V-5B | `model.wan.checkpoint_path` / `vae_path` / `config_path`; `t5.checkpoint_path` / `t5.tokenizer_path` (eval); `--wan_path` (RoboTwin deploy) | Must include `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth` and `google/umt5-xxl` |
| Qwen3-VL-2B-Instruct | `model.vlm.checkpoint_path` | Also used by the training datasets to build VLM inputs; passed as `vlm_path` to the RoboTwin deploy policy |

---

## RoboTwin

The RoboTwin pipeline finetunes EWAM on RoboTwin2.0 demonstrations and evaluates through the benchmark's standard policy interface. `inference/robotwin/deploy_policy.py` implements the RoboTwin 2.0 policy contract (`encode_obs` / `get_model` / `eval` / `reset_model`): it rebuilds the model from the checkpoint, encodes instructions with the Wan2.2 T5 encoder, assembles the T-shape observation the same way as training, pads the 14-dim qpos state to 16 dims, and restores the 16-dim action chunk back to 14-dim qpos for execution.

### Setup

The [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) benchmark (SAPIEN-based) is a third-party dependency and runs in its **own conda environment** (separate from training — RoboTwin/sapien + a torch stack, see `inference/robotwin/requirements.txt`). Clone the official repository and follow its install guide (conda env + sapien):

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
# install guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

All EWAM RoboTwin results were produced against RoboTwin 2.0 (`script/eval_policy.py`) on the official `main` branch.

### Data preparation

The training dataset (`robotwin_dim16`) expects, per task:

```
dataset_dir/
├── clean/<task_name>/              # clean demonstrations
│   ├── qpos/<episode>.pt           # [T, 14] robot states per frame (see below)
│   ├── videos/<episode>.mp4        # T-shape camera video, decoded with decord
│   ├── metas/<episode>.txt         # instruction text (with scene prefix)
│   └── umt5_wan/<episode>.pt       # list of pre-encoded umt5-xxl embeddings
└── randomized/<task_name>/         # randomized demonstrations (same layout)
```

- **qpos**: a `[T, 14]` tensor — dual-arm layout `[left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]`. The dataset unifies it to 16 dims at load time: `[left_joints(6), pad(1), left_gripper(1), right_joints(6), pad(1), right_gripper(1)]`. Other dims (7 / 8 / n≤16) are auto-padded; other robots can be added by following the same unification rule in `robotwin_agilex_dataset_dim16.py`.
- **videos**: three camera views (head / left / right) concatenated in a **T-shape** layout (head on top, left|right below), resized to `(video_height, video_width)` = `(384, 320)` from the training config at load time.
- **umt5_wan**: a list of tensors `[seq_len, 4096]` (one per instruction variant); one is sampled randomly per `__getitem__`.
- **Normalization**: none — RoboTwin training uses raw state/action values.

#### Converting RoboTwin2.0 (recommended path)

`data/robotwin2/robotwin_data_convert/` converts the official RoboTwin2.0 release into exactly the layout above, including T5 embeddings and the scene prefix in `metas/`:

```bash
cd data/robotwin2/robotwin_data_convert

# 1) Download from HuggingFace (set HF_ENDPOINT=https://hf-mirror.com to use a mirror)
python3 download_robotwin_dataset.py --output_dir /path/to/robotwin_raw_dataset

# 2) Edit config.yml: source_root / target_root / wan_repo_path
vim config.yml

# 3) Convert: hdf5 → videos + qpos + metas, then encode umt5-xxl embeddings
./run_conversion.sh
```

See `data/robotwin2/robotwin_data_convert/README.md` for details (task selection, worker counts, T5 GPU config). If the downloaded archives unpack with a duplicated directory level, `fix_duplicate_paths.py` repairs the layout.

#### Custom data

To train on your own robot, produce the four directories above per task. The minimal contract enforced by the loader is: every episode needs `qpos/<episode>.pt` + `videos/<episode>.mp4` + `umt5_wan/<episode>.pt`; `metas/` provides the text used for VLM inputs.

### Training

Training uses the shared entry point `train/train_ewam.py`, launched with `configs/ewam_robotwin.yaml` by `scripts/train_ewam_robotwin.sh` (torchrun + DeepSpeed ZeRO-1). By default it stage-2-finetunes from the released stage-1 pretrain checkpoint (`finetune.checkpoint_path` — the two init modes are described under [Pretrained weights](#2-pretrained-weights)). `model.loss_reweight: true` selects the reweighted loss (Gaussian timestep weighting + shifted timestep sampling) inside the shared model class; the plain variant (used by LIBERO) shares the same state dict and inference path.

```bash
bash scripts/train_ewam_robotwin.sh
# Override GPUs / port:
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29101 bash scripts/train_ewam_robotwin.sh
# Multi-node:
NNODES=2 RANK=0 MASTER_ADDR=<ip> bash scripts/train_ewam_robotwin.sh
```

**Released RoboTwin checkpoints** (download from [HuggingFace](https://huggingface.co/HaoWang00/EWAM)): **in-domain** — 32 GPUs, per-GPU batch 8 (effective 256), 60,000 steps; **c2r** — 16 GPUs, per-GPU batch 8 (effective 128), 20,000 steps.

Key settings in `configs/ewam_robotwin.yaml`:

- `common`: `action_dim=16` / `state_dim=16`, `num_video_frames=8`, `video_height=384`, `video_width=320`, `video_action_freq_ratio=2` → action chunk = 16.
- `dataset`: `type: robotwin_dim16`, `dataset_dir`, `data_mode: both` (clean + randomized), `task_mode: multi`.
- `training`: AdamW, lr `5e-5` (separate groups for main / WAN / VLM), grad clip 0.5, bf16, DeepSpeed ZeRO-1 (`configs/zero1.json`).
- `model.loss_weights`: video 1.0 / action 1.0; `model.loss_reweight: true` selects the reweighted loss (Gaussian timestep weighting) inside the model.

Checkpoints are written under `system.checkpoint_dir / <run name>/`:

- `checkpoint_step_N/` every `system.save_interval` steps (Accelerate/DeepSpeed full state);
- `best_action_l2/` — lowest validation action-L2 (`system.val_interval`);
- `config.json` next to the weights for reproducibility.

(RoboTwin training uses raw state/action values, so no normalization stats are written.)

Resume:

```yaml
resume:
  checkpoint_path: /path/to/checkpoint_step_40000   # restores optimizer/scheduler state
```

### Evaluation

1. **Deploy the policy directory** — copy everything under `inference/robotwin/` into `RoboTwin/policy/Ewam_dim16/`:

```bash
mkdir -p /path/to/RoboTwin/policy/Ewam_dim16
cp inference/robotwin/{deploy_policy.py,deploy_policy.yml,auto_eval.sh,eval.sh,paths_config.example.yml,requirements.txt,tasks_all.txt} \
   /path/to/RoboTwin/policy/Ewam_dim16/
# alternative (keeps the deployed copy in sync with this repo):
# ln -sfn "$(pwd)/inference/robotwin" /path/to/RoboTwin/policy/Ewam_dim16
```

2. **Create `paths_config.yml`** from the example and fill in the paths:

```bash
cd /path/to/RoboTwin/policy/Ewam_dim16
cp paths_config.example.yml paths_config.yml
vim paths_config.yml
```

| Key | Meaning |
|---|---|
| `robotwin_root` | RoboTwin checkout root (contains `script/eval_policy.py`) |
| `conda_env` | RoboTwin conda environment name |
| `checkpoint_path` | trained checkpoint — the `checkpoint_step_N/pytorch_model/` directory (containing `mp_rank_00_model_states.pt`), or the `.pt` file itself; download the released RoboTwin (c2r / in-domain) checkpoint from [HaoWang00/EWAM](https://huggingface.co/HaoWang00/EWAM) |
| `ewam_root` | this EWAM repository — the policy imports `models/`, `wan/`, `utils/`, `data/` from here |
| `wan_path` / `vlm_path` | Wan2.2-TI2V-5B / Qwen3-VL-2B-Instruct (config/tokenizer only, weights come from the checkpoint) |
| `config_path` *(optional)* | training config yaml that produced the checkpoint; defaults to `<ewam_root>/configs/ewam_robotwin.yaml` |
| `gpu_ids` *(optional)* | empty `[]` = auto-detect all GPUs |
| `task_config` / `seed` / `tasks_file` | `demo_randomized` / `42` / `tasks_all.txt` (all 50 tasks) |

3. **Install the inference dependencies into the RoboTwin env**:

```bash
conda activate RoboTwin
cd /path/to/RoboTwin/policy/Ewam_dim16
pip install -r requirements.txt
```

**Important**: `config_path` must be the training config that produced the checkpoint (architecture, image layout 384×320, dims). The scene prefix added to instructions is fixed to match the training data converter — if you trained on custom data with a different prefix, update `SCENE_PREFIX` in `deploy_policy.py`.

For manual (script-free) integration, the same values can be passed as `usr_args` keys (`ckpt_setting`, `wan_path`, `vlm_path`, `config_path`) or environment variables (`EWAM_ROOT`, `EWAM_WAN_PATH`, `EWAM_VLM_PATH`, `EWAM_CONFIG`).

#### Running

```bash
cd /path/to/RoboTwin
bash policy/Ewam_dim16/eval.sh         # single task (edit TASK_NAME / GPU_ID inside)
bash policy/Ewam_dim16/auto_eval.sh    # all 50 tasks, parallel across GPUs
```

Both scripts wrap RoboTwin's own entry point:

```bash
python script/eval_policy.py \
    --config policy/Ewam_dim16/deploy_policy.yml \
    --overrides \
    --task_name <task> --task_config demo_randomized \
    --ckpt_setting /path/to/checkpoint_step_N/pytorch_model \
    --seed 42 --policy_name Ewam_dim16 --log_dir <log_dir> \
    --wan_path /path/to/Wan2.2-TI2V-5B --vlm_path /path/to/Qwen3-VL-2B-Instruct
```

Outputs: per-task stdout under `eval_logs/.../<task>.log`; per-task success rates written by RoboTwin under `eval_result/<task>/<policy>/<task_config>/.../_result.txt`; `auto_eval.sh` finishes with `evaluation_summary.txt` across all tasks. Per-episode frame-grid visualizations (condition | predicted frames) are saved under `LOG_DIR/images/<task_name>/` when logging is enabled.

Instructions are sampled as **unseen** (`instruction_type: unseen` in `deploy_policy.yml`), To evaluate with seen instructions instead, add `instruction_type=seen` to the `--overrides` list (typically worth one or two points).

---

## LIBERO

The LIBERO pipeline finetunes EWAM on LIBERO demonstrations (LeRobot format) and evaluates across the four LIBERO suites with the dim16 horizontal variant.

### Setup

The LIBERO benchmark is a third-party dependency and is **not** pip-installable. Clone it and expose it via `LIBERO_ROOT` (the eval launcher adds it to `PYTHONPATH` automatically):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git <path-to>/LIBERO-master
export LIBERO_ROOT=<path-to>/LIBERO-master
```

Install the simulation stack **into the same environment you run EWAM in** — these are what `experiments/libero/libero_utils.py` imports at rollout time:

```bash
pip install "robosuite==1.4.0" "mujoco==3.3.2" bddl "gym==0.25.2" easydict cloudpickle
```

> Do **not** run `pip install -r $LIBERO_ROOT/requirements.txt`: its pins (numpy==1.22.4, transformers==4.21.1, hydra-core==1.2.0) are older than what EWAM needs and will break the environment. Only the simulation packages above are required at eval time. Keep `mujoco` consistent with the LIBERO data version (3.3.2 for our released datasets).

Headless rendering additionally needs system libraries (the eval launcher sets `MUJOCO_GL=osmesa`):

```bash
sudo apt-get install -y libosmesa6-dev patchelf libgl1 libglfw3
```

### Data preparation

The training dataset (`libero_lerobot`) expects one **LeRobot-format** directory per suite (produced by the standard LIBERO → LeRobot conversion tools):

```
libero_<suite>_no_noops_lerobot/
├── data/chunk-000/episode_XXXXXX.parquet       # state (8-D EEF) + action (7-D OSC delta) + task_index
├── videos/chunk-000/
│   ├── observation.images.image/episode_XXXXXX.mp4        # agentview camera
│   └── observation.images.wrist_image/episode_XXXXXX.mp4  # wrist camera
├── meta/tasks.jsonl                            # task instructions
├── meta/stat.json                              # normalization statistics (min/max)
└── t5_embedding/episode_XXXXXX.pt              # pre-encoded umt5-xxl embeddings (see below)
```

Processing (done inside `libero_lerobot_dataset.py`): state is 8-D EEF `[pos(3), axis-angle(3), grip(2)]`, action is 7-D OSC delta (position/rotation delta, gripper absolute), min/max normalization to `[-1, 1]` from `meta/stat.json`, agentview | wrist concatenated **horizontally** at 224×448, then both are padded to 16 dims (`unify_to_16: true`).

After conversion, generate the T5 embeddings for each suite directory (uses the Wan2.2 T5 encoder and the LIBERO task prefix that matches training):

```bash
cd <EWAM root>
python data/libero/regenerate_libero_t5_with_prefix.py \
    --dataset_dir /path/to/libero_object_no_noops_lerobot \
    --wan_path /path/to/Wan2.2-TI2V-5B
# repeat for libero_10 / libero_goal / libero_spatial
```

### Training

Training uses the shared entry point `train/train_ewam.py`, launched with `configs/ewam_libero.yaml` by `scripts/train_ewam_libero.sh` (torchrun + DeepSpeed ZeRO-1). `model.loss_reweight: false` here — LIBERO trains with the plain uniform-mean-MSE flow-matching loss.

Edit `configs/ewam_libero.yaml`:

- `dataset.dataset_dirs`: the four suite directories;
- `model.wan.*` / `model.vlm.checkpoint_path`: backbone weight paths;
- `finetune.checkpoint_path`: stage-1 pretrain checkpoint (released on [HuggingFace](https://huggingface.co/HaoWang00/EWAM), see [Pretrained weights](#2-pretrained-weights); or `null` for from-scratch);
- `system.checkpoint_dir` / `logging.tensorboard_log_dir`: output locations.

```bash
bash scripts/train_ewam_libero.sh
# Override GPUs / port:
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29201 bash scripts/train_ewam_libero.sh
```

**The released LIBERO checkpoint** (download from [HuggingFace](https://huggingface.co/HaoWang00/EWAM)): 8 GPUs, per-GPU batch 8 (effective 64), 20,000 steps.

Key settings: `action_dim=16` / `state_dim=16` (padded from LIBERO 7/8), horizontal 224×448 images, action chunk = 16, `image_concat_mode: horizontal`, `unify_to_16: true` (the padded dims are masked out of the loss via `action_dim_is_pad`), `model.loss_reweight: false` (plain uniform-MSE loss).

Checkpoints are written under `system.checkpoint_dir / <run name>/`:

- `checkpoint_step_N/` every `system.save_interval` steps (Accelerate/DeepSpeed full state);
- `best_action_l2/` — lowest validation action-L2 (`system.val_interval`);
- `config.json` next to the weights for reproducibility;
- `dataset_stats.json` (over the RAW action/state dims) — written at startup; the LIBERO eval picks it up automatically.

Resume:

```yaml
resume:
  checkpoint_path: /path/to/checkpoint_step_40000   # restores optimizer/scheduler state
```

### Evaluation

#### Quick start

```bash
export LIBERO_ROOT=/path/to/LIBERO-master        # from LIBERO Setup above
bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
# Override checkpoint / stats / GPUs / trials (the released LIBERO checkpoint is on
# https://huggingface.co/HaoWang00/EWAM — pass its mp_rank_00_model_states.pt via CKPT):
CKPT=/path/to/checkpoint_step_40000/pytorch_model/mp_rank_00_model_states.pt \
STATS=/path/to/dataset_stats.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_TRIALS=50 \
  bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
# Single suite / fewer trials:
NUM_TRIALS=10 SUITES="libero_object" bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
```

Environment variables: `CKPT`, `STATS`, `SUITES`, `NUM_TRIALS`, `NUM_TASKS_PER_SUITE`, `CUDA_VISIBLE_DEVICES` / `NUM_GPUS`, `MAX_TASKS_PER_GPU` (concurrent tasks per GPU, default 3), `OUTPUT_DIR`, `LIBERO_ROOT`, `CONDA_ENV`.

The launcher fills a tmux pane pool (each GPU hosts up to `MAX_TASKS_PER_GPU` single-process evals), dispatches tasks to the least-loaded GPU, aborts on the first failed task, and finally prints a per-suite success-rate table plus `summary.csv`.

#### Single-task debugging

```bash
export PYTHONPATH="$PWD:$LIBERO_ROOT"
export MUJOCO_GL=osmesa HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_ewam_libero_single_horizontal_dim16.py \
    ckpt=/path/to/mp_rank_00_model_states.pt \
    EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
    EVALUATION.num_trials=5 EVALUATION.output_dir=./evaluate_results/debug
```

#### What the eval does

- **dim16 conversions**: LIBERO's 7-dim actions are restored from the padded 16-dim chunk before denormalization (`dataset.unify_to_16: true`); the 8-dim state is padded to 16 before inference.
- **Images**: agentview | wrist concatenated horizontally at 224×448 (`dataset.image_concat_mode: horizontal`).
- **Normalization stats** are resolved in order:
  1. `EVALUATION.dataset_stats_path` — the aggregated `dataset_stats.json` written by LIBERO training (or released alongside a trained checkpoint);
  2. `dataset_stats.json` found next to the checkpoint (up to 4 directory levels up);
  3. element-wise min/max aggregation of `meta/stat.json` across `dataset.dataset_dirs`;
  4. default `[-1, 1]` (a warning is logged — check the stats resolution before trusting results).

Outputs per run directory: `<suite>/task<k>_results.json` (per-task success/trials/duration), `summary.csv`, and per-task rollout MP4s.

---

## FAQ

**Q: Which paths must I edit before running?**
RoboTwin training: `configs/ewam_robotwin.yaml` (`dataset_dir`, `model.wan.*`, `model.vlm.checkpoint_path`, `finetune.checkpoint_path`, `system.checkpoint_dir`, `logging.tensorboard_log_dir`). LIBERO training: the same fields in `configs/ewam_libero.yaml`. LIBERO eval: `configs/ewam_libero_eval_horizontal_dim16.yaml` (same weight fields + `EVALUATION.dataset_stats_path`). RoboTwin eval: `inference/robotwin/paths_config.yml` (copy from `paths_config.example.yml`). Data conversion: `data/robotwin2/robotwin_data_convert/config.yml`, and `LIBERO_ROOT` if your LIBERO checkout lives elsewhere.

**Q: Eval warns "No dataset_stats.json or stat.json found, using default [-1, 1]"?**
Stats resolution failed — results will be misnormalized. Provide `STATS=` pointing at the stats file written during LIBERO training (or released with the checkpoint), or point `dataset.dataset_dirs` at the LIBERO lerobot datasets (their `meta/stat.json` files are aggregated automatically). RoboTwin training/eval does not use stats (raw values).

**Q: RoboTwin deploy says "EWAM repository root not found"?**
The deployed policy (`RoboTwin/policy/Ewam_dim16/deploy_policy.py`) cannot see the EWAM package. Set `ewam_root` in `paths_config.yml` (the eval scripts export it as `EWAM_ROOT` automatically), or `export EWAM_ROOT=/path/to/EWAM`.

**Q: Can RoboTwin eval share the training conda environment?**
No. RoboTwin is SAPIEN-based and needs its own environment (see its install guide). Install `inference/robotwin/requirements.txt` **into the RoboTwin env** after RoboTwin itself is set up; keep the training env untouched. LIBERO eval, by contrast, runs inside the EWAM environment (with the extra simulation packages from the LIBERO Setup section).

**Q: LIBERO training crashes with missing `t5_embedding/episode_XXXXXX.pt`?**
Generate the T5 embeddings for every suite directory first: `python data/libero/regenerate_libero_t5_with_prefix.py --dataset_dir <suite_dir> --wan_path <Wan2.2-TI2V-5B>`.

**Q: MuJoCo fails to render on a headless node?**
Keep `MUJOCO_GL=osmesa` (already set by the launcher). If osmesa is unavailable, `export MUJOCO_GL=egl` on GPU nodes.

**Q: CUDA OOM during training?**
Lower `training.batch_size`, or enable `training.gradient_accumulation_steps > 1` (effective batch = batch_size × grad-accum × num GPUs). The video model dominates memory; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set by the entry script.

## Acknowledgements

- [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Alibaba) — video backbone, VAE and T5 modules (vendored under `wan/`).
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — vision-language model.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — simulation benchmark.
- [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) — deployment benchmark and training data source.
- [Motus](https://github.com/thu-ml/Motus) — the RoboTwin data conversion and evaluation integration reference Motus.
- [FastWAM](https://github.com/yuantianyuan01/FastWAM) — LIBERO evaluation protocol and parallel scheduling lineage.

## License

This repository is released under the [Apache License 2.0](LICENSE). The vendored `wan/` code (Wan2.2, © Alibaba) retains its original Apache 2.0 copyright headers.
