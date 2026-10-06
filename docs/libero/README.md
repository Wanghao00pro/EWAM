# LIBERO Guide

The LIBERO pipeline finetunes EWAM on LIBERO demonstrations (LeRobot format) and evaluates across the four LIBERO suites with the dim16 horizontal variant.

## Setup

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

## Data preparation

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

## Training

Training uses the shared entry point `train/train_ewam.py`, launched with `configs/ewam_libero.yaml` by `scripts/train_ewam_libero.sh` (torchrun + DeepSpeed ZeRO-1).

Edit `configs/ewam_libero.yaml`:

- `dataset.dataset_dirs`: the four suite directories;
- `model.wan.*` / `model.vlm.checkpoint_path`: backbone weight paths;
- `finetune.checkpoint_path`: stage-1 pretrain checkpoint (released on [HuggingFace](https://huggingface.co/HaoWang00/EWAM), see [Model Weights and Configuration](../model_weights.md); or `null` for from-scratch);
- `system.checkpoint_dir` / `logging.tensorboard_log_dir`: output locations.

```bash
bash scripts/train_ewam_libero.sh
# Override GPUs / port:
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29201 bash scripts/train_ewam_libero.sh
```

**The released LIBERO checkpoint** (download from [HuggingFace](https://huggingface.co/HaoWang00/EWAM)): 8 GPUs, per-GPU batch 8 (effective 64), 20,000 steps.

Key settings: `action_dim=16` / `state_dim=16` (padded from LIBERO 7/8), horizontal 224×448 images, action chunk = 16, `image_concat_mode: horizontal`, `unify_to_16: true` (the padded dims are masked out of the loss via `action_dim_is_pad`).

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

## Evaluation

### Quick start

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

### Single-task debugging

```bash
export PYTHONPATH="$PWD:$LIBERO_ROOT"
export MUJOCO_GL=osmesa HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_ewam_libero_single_horizontal_dim16.py \
    ckpt=/path/to/mp_rank_00_model_states.pt \
    EVALUATION.task_suite_name=libero_object EVALUATION.task_id=0 \
    EVALUATION.num_trials=5 EVALUATION.output_dir=./evaluate_results/debug
```

### What the eval does

- **dim16 conversions**: LIBERO's 7-dim actions are restored from the padded 16-dim chunk before denormalization (`dataset.unify_to_16: true`); the 8-dim state is padded to 16 before inference.
- **Images**: agentview | wrist concatenated horizontally at 224×448 (`dataset.image_concat_mode: horizontal`).
- **Normalization stats** are resolved in order:
  1. `EVALUATION.dataset_stats_path` — the aggregated `dataset_stats.json` written by LIBERO training (or released alongside a trained checkpoint);
  2. `dataset_stats.json` found next to the checkpoint (up to 4 directory levels up);
  3. element-wise min/max aggregation of `meta/stat.json` across `dataset.dataset_dirs`;
  4. default `[-1, 1]` (a warning is logged — check the stats resolution before trusting results).

Outputs per run directory: `<suite>/task<k>_results.json` (per-task success/trials/duration), `summary.csv`, and per-task rollout MP4s.
