# RoboTwin 2.0 Guide

The RoboTwin pipeline finetunes EWAM on RoboTwin2.0 demonstrations and evaluates through the benchmark's standard policy interface. `inference/robotwin/deploy_policy.py` implements the RoboTwin 2.0 policy contract (`encode_obs` / `get_model` / `eval` / `reset_model`): it rebuilds the model from the checkpoint, encodes instructions with the Wan2.2 T5 encoder, assembles the T-shape observation the same way as training, pads the 14-dim qpos state to 16 dims, and restores the 16-dim action chunk back to 14-dim qpos for execution.

## Setup

The [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) benchmark (SAPIEN-based) is a third-party dependency and runs in its **own conda environment** (separate from training — RoboTwin/sapien + a torch stack, see `inference/robotwin/requirements.txt`). Clone the official repository and follow its install guide (conda env + sapien):

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
# install guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

All EWAM RoboTwin results were produced against RoboTwin 2.0 (`script/eval_policy.py`) on the official `main` branch.

## Data preparation

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

### Converting RoboTwin2.0 (recommended path)

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

### Custom data

To train on your own robot, produce the four directories above per task. The minimal contract enforced by the loader is: every episode needs `qpos/<episode>.pt` + `videos/<episode>.mp4` + `umt5_wan/<episode>.pt`; `metas/` provides the text used for VLM inputs.

## Training

Training uses the shared entry point `train/train_ewam.py`, launched with `configs/ewam_robotwin.yaml` by `scripts/train_ewam_robotwin.sh` (torchrun + DeepSpeed ZeRO-1). By default it stage-2-finetunes from the released stage-1 pretrain checkpoint (`finetune.checkpoint_path` — the two init modes are described under [Model Weights and Configuration](../model_weights.md)). `model.loss_reweight: true` selects the reweighted loss (Gaussian timestep weighting + shifted timestep sampling) inside the shared model class.

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

## Evaluation

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

### Running

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
