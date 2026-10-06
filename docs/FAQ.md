# FAQ

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
Lower `training.batch_size`, or enable `training.gradient_accumulation_steps > 1` (effective batch = batch_size × grad-accum × num GPUs).
