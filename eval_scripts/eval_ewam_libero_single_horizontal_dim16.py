#!/usr/bin/env python3
"""
EWAM LIBERO Evaluation Script (Single-GPU) — dim16 (Horizontal + UnifyTo16)

T-shape variant lineage, adapted for the
horizontal finetune (configs/libero_lerobot_wan_vlm_mask_horizontal_dim16.yaml):
  - state_dim=16 / action_dim=16 (UnifyTo16 padded, matching pretrain/finetune ckpt)
  - image: horizontal concat (agentview | wrist side-by-side), 224x448
  - normalization: min/max [-1,1], stats aggregated across ALL 4 suites (element-wise
    min of mins / max of maxs), matching the training-side _load_stats aggregation.

dim16 conversions at eval time (UnifyTo16 layout / inverse):
  - State  8 -> 16: [eef(6), grip_open@6, grip_close@7, pad@8-15]  (pad before inference)
  - Action 16 -> 7: take idx [0..5, 7] = [eef(6), grip]            (restore before denormalize)

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_ewam_libero_single_horizontal_dim16.py \
        ckpt=./checkpoints/... \
        EVALUATION.task_suite_name=libero_object \
        EVALUATION.task_id=0 \
        EVALUATION.num_trials=10 \
        EVALUATION.output_dir=./evaluate_results/ewam_libero_horizontal_dim16
"""

# Task prefix for VLM and T5 encoding (must match training).
TASK_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: a front camera, a wrist camera and its copy. "
    "The Franka robot arm is currently performing the following task: "
)

# Default fallback: aggregate across all 4 LIBERO suites' meta/stat.json ).
_DEFAULT_DATASET_DIRS = [
    "/path/to/libero_lerobot/libero_object_no_noops_lerobot",
    "/path/to/libero_lerobot/libero_10_no_noops_lerobot",
    "/path/to/libero_lerobot/libero_goal_no_noops_lerobot",
    "/path/to/libero_lerobot/libero_spatial_no_noops_lerobot",
]

import json
import logging
import os
import sys
import time
import atexit
from pathlib import Path
from typing import Any, Optional, List

import hydra
import numpy as np
import torch
from accelerate import PartialState
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Attention analysis setup
_ATTENTION_ANALYSIS_DIR = os.environ.get("ATTENTION_ANALYSIS_DIR", "")
_ATTENTION_MAX_STEPS = int(os.environ.get("ATTENTION_ANALYSIS_MAX_STEPS", "0"))
_attention_capture = None

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from libero.libero import benchmark

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    return torch.bfloat16


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to (height, width) using bilinear interpolation.

    This matches training's torch.nn.functional.interpolate(mode='bilinear'),
    without the center-crop step that _center_crop_resize performs.
    Training resizes each camera directly to target size, then concatenates.
    """
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build 8D EEF state from LIBERO observation.

    Returns:
        state: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
    """
    state = np.concatenate((
        obs["robot0_eef_pos"],                    # 3 dims
        quat2axisangle(obs["robot0_eef_quat"]),   # 3 dims
        obs["robot0_gripper_qpos"],               # 2 dims
    )).astype(np.float32)  # Total: 8 dims
    return state


def _pad_state_to_16(state: torch.Tensor) -> torch.Tensor:
    """Pad 8-D LIBERO state to 16-D UnifyTo16 layout (UnifyTo16 layout, dim=8).

    [eef(6), grip_open@6, grip_close@7, pad@8-15]
      input state = [eef_pos(3), axis_angle(3), grip_open, grip_close]  (8-D)
      -> left=state[:7] (eef+grip_open), grip=state[7:8] (grip_close), pad=zeros(8)
    """
    assert state.shape[-1] == 8, f"Expected 8-D state, got {state.shape[-1]}-D"
    left = state[..., :7]                                       # idx 0-6
    grip = state[..., 7:8]                                      # idx 7
    pad = torch.zeros(*state.shape[:-1], 8, dtype=state.dtype, device=state.device)  # idx 8-15
    return torch.cat([left, grip, pad], dim=-1)                # 7+1+8 = 16


def _restore_action_to_7(action: np.ndarray) -> np.ndarray:
    """Restore 16-D UnifyTo16 action to 7-D LIBERO (UnifyTo16 inverse, orig_dim=7).

    16-D layout: [eef(6), pad@6, grip@7, pad@8-15] -> 7-D [eef(6), grip] = idx [0..5, 7].
    """
    assert action.shape[-1] == 16, f"Expected 16-D action, got {action.shape[-1]}-D"
    left = action[..., :6]       # eef (idx 0-5)
    grip = action[..., 7:8]      # grip (idx 7)
    return np.concatenate([left, grip], axis=-1)  # 6+1 = 7


def _load_stat(stat_path: str) -> dict:
    """Load normalization statistics from a single dataset_stats.json or stat.json.

    Supports two formats:
    1. Format: {state: {default: {global_min, global_max, ...}}, action: {default: {...}}}
    2. Flat format: {observation.state: {min, max}, action: {min, max}}

    Always returns flat format: {observation.state: {min, max}, action: {min, max}}
    """
    with open(stat_path, "r") as f:
        raw = json.load(f)

    if "state" in raw and isinstance(raw["state"], dict) and "default" in raw["state"]:
        flat = {}
        for src_key, dst_key in [("state", "observation.state"), ("action", "action")]:
            if src_key in raw and "default" in raw[src_key]:
                entry = raw[src_key]["default"]
                mins = entry.get("global_min", entry.get("min"))
                maxs = entry.get("global_max", entry.get("max"))
                if mins is not None and maxs is not None:
                    flat[dst_key] = {"min": mins, "max": maxs}
        return flat
    return raw


def _load_normalization_stat(cfg: DictConfig) -> dict:
    """Load normalization stats, matching the training-side aggregation.

    Search order:
    1. Explicit cfg.EVALUATION.dataset_stats.json (single aggregated file saved by training)
    2. cfg.ckpt parent directories (up to 4 levels) for dataset_stats.json
    3. Aggregate meta/stat.json across ALL cfg.dataset.dataset_dirs (element-wise min of mins,
       max of maxs) — Falls back to _DEFAULT_DATASET_DIRS.
    4. Hardcoded default [-1, 1]

    Returns flat format: {observation.state: {min, max}, action: {min, max}} (8-D state / 7-D action).
    """
    # 1. Explicit path from config
    candidates: list[Path] = []
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    # 2. Checkpoint parent directories (stats saved alongside checkpoints by training)
    ckpt = cfg.get("ckpt", None)
    if ckpt is not None:
        ckpt_path = Path(os.path.expanduser(os.path.expandvars(str(ckpt))))
        for parent in list(ckpt_path.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            logging.info(f"Loaded normalization stats (single file) from {resolved}")
            return _load_stat(str(resolved))

    # 3. Aggregate across all dataset_dirs (union range across suites)
    dataset_dirs = cfg.get("dataset", {}).get("dataset_dirs", None)
    if not dataset_dirs:
        dataset_dirs = _DEFAULT_DATASET_DIRS
        logging.info(f"No dataset_dirs in config, using default {len(dataset_dirs)} suites for stat aggregation")

    collected = {}  # key -> list of (mins, maxs) per suite
    for d in dataset_dirs:
        p = Path(d) / "meta" / "stat.json"
        if not p.exists():
            logging.warning(f"stat.json not found, skipping: {p}")
            continue
        flat = _load_stat(str(p))
        for key, entry in flat.items():
            if "min" in entry and "max" in entry:
                collected.setdefault(key, []).append((entry["min"], entry["max"]))
        logging.info(f"  stat source: {p}")

    if collected:
        aggregated = {}
        for key, entries in collected.items():
            mins_arr = [e[0] for e in entries]
            maxs_arr = [e[1] for e in entries]
            aggregated[key] = {
                "min": [min(col) for col in zip(*mins_arr)],
                "max": [max(col) for col in zip(*maxs_arr)],
            }
        n_suites = len(next(iter(collected.values())))
        logging.info(f"Aggregated normalization stats across {n_suites} suite(s) (element-wise min/max):")
        for key in aggregated:
            logging.info(f"  {key}: min={[round(v, 4) for v in aggregated[key]['min']]}")
            logging.info(f"  {key}: max={[round(v, 4) for v in aggregated[key]['max']]}")
        return aggregated

    # 4. Fallback default
    logging.warning("No dataset_stats.json or stat.json found, using default [-1, 1]")
    return {
        "observation.state": {"min": [-1] * 8, "max": [1] * 8},
        "action": {"min": [-1] * 7, "max": [1] * 7},
    }


def _normalize_state(state: np.ndarray, stat: dict) -> torch.Tensor:
    """Normalize 8-D state to [-1,1] using min-max (linear min-max normalizer).

    scale = (output_max - output_min) / (max - min)
    offset = output_min - scale * min
    normalized = state * scale + offset, clamped to [-5, 5]
    """
    mins = np.array(stat["observation.state"]["min"], dtype=np.float64)
    maxs = np.array(stat["observation.state"]["max"], dtype=np.float64)
    output_min, output_max = -1.0, 1.0

    ranges = maxs - mins
    ignore_dim = ranges < 1e-4
    ranges = np.where(ignore_dim, output_max - output_min, ranges)

    scale = (output_max - output_min) / ranges
    offset = output_min - scale * mins
    offset = np.where(ignore_dim, (output_max + output_min) / 2.0 - mins, offset)

    normalized = state.astype(np.float64) * scale + offset
    normalized = np.clip(normalized, -5.0, 5.0)
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)


def _denormalize_action(action: np.ndarray, stat: dict) -> np.ndarray:
    """Denormalize 7-D action from [-1,1] back to raw space (normalizer backward).

    backward: x = (x - offset) / scale
    """
    mins = np.array(stat["action"]["min"], dtype=np.float64)
    maxs = np.array(stat["action"]["max"], dtype=np.float64)
    output_min, output_max = -1.0, 1.0

    ranges = maxs - mins
    ignore_dim = ranges < 1e-4
    ranges = np.where(ignore_dim, output_max - output_min, ranges)

    scale = (output_max - output_min) / ranges
    offset = output_min - scale * mins
    offset = np.where(ignore_dim, (output_max + output_min) / 2.0 - mins, offset)

    return (action.astype(np.float64) - offset) / scale


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
    vlm_processor,
    task_description: str,
    stat: Optional[dict] = None,
):
    """Convert LIBERO observation to EWAM model input.

    EWAM: image in [0,1] range (does *2-1 internally).
    State: normalized 8-D -> [-1,1], then padded 8->16 (UnifyTo16) for the dim16 model.
    Image: horizontal concatenation (agentview | wrist side-by-side).
    """
    imgs = get_libero_image(obs)

    # Horizontal concatenation: agentview + wrist side by side.
    # Matches training dataset's _concat_horizontal:
    #   1. Resize each camera to [H, W]  (pure bilinear resize, NO center-crop — matches
    #      training's torch.nn.functional.interpolate; _center_crop_resize would diverge)
    #   2. Concat horizontally: [H, W] + [H, W] = [H, 2W]
    #   3. Final resize to (height, width) = [H, W]
    #   ┌────────────┬────────────┐
    #   │  agentview │   wrist    │  H x (W + W) = H x 2W
    #   └────────────┴────────────┘
    #   Final: resize to (height, width)
    agent_rgb = _resize(imgs["image"], width=width, height=height)
    wrist_rgb = _resize(imgs["wrist_image"], width=width, height=height)

    rgb = np.hstack([agent_rgb, wrist_rgb])  # (H, 2W, 3)

    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = np.array(Image.fromarray(rgb).resize((width, height), resample=Image.BILINEAR))

    # EWAM: Keep [0,1] range (model does *2-1 internally)
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x / 255.0  # [0,1] range for EWAM

    # State: normalize 8-D if use_normalization=true, else raw 8-D
    state = _extract_sim_state(obs)
    use_norm = bool(cfg.get("dataset", {}).get("use_normalization", False))
    if use_norm and stat is not None:
        state_t = _normalize_state(state, stat).to(device=device)
        logging.debug(f"[NORM] Raw state: {state[:3]}... -> Normalized: {state_t[0, :3].cpu().numpy()}...")
    else:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device=device)

    # Pad state 8 -> 16 (UnifyTo16) for the dim16 model
    state_dim = int(cfg.common.get("state_dim", 16))
    if state_dim == 16:
        state_t = _pad_state_to_16(state_t)
    logging.debug(f"[STATE] state shape to model: {tuple(state_t.shape)}")

    # Build VLM inputs for understanding expert
    prefixed_task = TASK_PREFIX + task_description
    img_pil = Image.fromarray(rgb.astype(np.uint8))
    vlm_inputs = _build_vlm_inputs(vlm_processor, prefixed_task, img_pil, device)

    return x, state_t, vlm_inputs, imgs


def _build_vlm_inputs(processor, text_instruction: str, image_pil: Image.Image, device: str):
    """Build VLM inputs for EWAM understanding expert."""
    from utils.vlm_utils import preprocess_vlm_messages
    vlm_inputs = preprocess_vlm_messages(text_instruction, image_pil, processor)
    if vlm_inputs:
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in vlm_inputs.items()}
    return vlm_inputs


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    vlm_processor,
    t5_encoder,
    stat: Optional[dict] = None,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict]:
    """Run EWAM inference to predict action chunk.

    dim16: model outputs [chunk, 16] (UnifyTo16). Restore to 7-D [eef(6), grip] BEFORE
    denormalizing (stats are over the raw 7-D action).
    """
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 20))
    use_norm = bool(cfg.get("dataset", {}).get("use_normalization", False))
    action_dim = int(cfg.common.get("action_dim", 16))

    # Get image, state (padded to 16), and VLM inputs
    image, state, vlm_inputs, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.dtype,
        vlm_processor=vlm_processor,
        task_description=task_description,
        stat=stat,
    )

    # Get T5 language embeddings (use prefixed task to match training)
    prefixed_task = TASK_PREFIX + task_description
    language_embeddings = None
    if t5_encoder is not None:
        try:
            t5_out = t5_encoder([prefixed_task], model_device)
            if isinstance(t5_out, torch.Tensor):
                language_embeddings = [t5_out.squeeze(0)]
            else:
                language_embeddings = t5_out
        except Exception as e:
            logging.warning(f"T5 encoding failed: {e}")

    with torch.no_grad():
        # EWAM inference_step returns (predicted_frames, predicted_actions)
        # predicted_actions: [B, action_chunk_size, action_dim]  (action_dim=16 for dim16 model)
        model_to_use = model.module if hasattr(model, 'module') else model
        if _attention_capture is not None:
            with _attention_capture:
                predicted_frames, predicted_actions = model_to_use.inference_step(
                    first_frame=image,
                    state=state,
                    num_inference_steps=num_inference_steps,
                    language_embeddings=language_embeddings,
                    vlm_inputs=[vlm_inputs] if vlm_inputs else None,
                )
        else:
            predicted_frames, predicted_actions = model_to_use.inference_step(
                first_frame=image,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=[vlm_inputs] if vlm_inputs else None,
            )

    # predicted_actions: [B, chunk_size, action_dim]
    action = predicted_actions.cpu().float().numpy()[0]  # [chunk_size, action_dim]

    # dim16: restore 16-D UnifyTo16 -> 7-D [eef(6), grip] BEFORE denormalize
    if action_dim == 16:
        action = _restore_action_to_7(action)  # [chunk_size, 7]
    logging.debug(f"[ACTION] restored to 7-D, shape: {action.shape}")

    # Take first action_horizon steps
    action = action[:action_horizon]

    # DIAGNOSTIC: Print raw model output values (disabled — too noisy in batch eval)
    # logging.warning(f"[DIAG] Raw model output (7-D) - min: {action.min():.6f}, max: {action.max():.6f}, mean: {action.mean():.6f}")
    # logging.warning(f"[DIAG] First action sample: {action[0]}")
    # logging.warning(f"[DIAG] Gripper values (raw): {action[:, -1]}")

    # Action post-processing depends on normalization mode
    if use_norm and stat is not None:
        # Denormalize: [-1,1] -> raw space ), stats are over 7-D action
        action = _denormalize_action(action, stat)
        # logging.warning(f"[DIAG] After denormalize - min: {action.min():.6f}, max: {action.max():.6f}, mean: {action.mean():.6f}")
        # logging.warning(f"[DIAG] First action (denorm): {action[0]}")

        # Gripper handling:
        # After denorm, gripper is in raw [0,1] where 0=close, 1=open
        # LIBERO expects: -1=open, +1=close
        # So: [0,1] -> [-1,1] via *2-1, then invert sign
        action[..., -1] = action[..., -1] * 2 - 1  # [0,1] -> [-1,1]
        action = invert_gripper_action(action)        # invert sign
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])
    else:
        # No normalization: raw action space
        # Gripper: raw > 0.5 -> -1 (open), else -> +1 (close)
        action[..., -1] = np.where(action[..., -1] > 0.5, -1.0, 1.0)
        if bool(cfg.EVALUATION.get("binarize_gripper", False)):
            action[..., -1] = np.sign(action[..., -1])

    # logging.warning(f"[DIAG] Final action (first step): {action[0]}")
    # logging.warning(f"[DIAG] Final gripper: {action[:, -1]}")

    # Convert predicted_frames to numpy for saving
    if predicted_frames.dim() == 5:
        if predicted_frames.shape[1] == 3:  # [B, C, T, H, W]
            pred_frames_np = predicted_frames[0].permute(1, 0, 2, 3).cpu().numpy()
        else:  # [B, T, C, H, W]
            pred_frames_np = predicted_frames[0].cpu().numpy()
    else:
        pred_frames_np = predicted_frames.cpu().numpy()

    return action, imgs, pred_frames_np


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    vlm_processor,
    t5_encoder,
    episode_idx: int,
    stat: Optional[dict] = None,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    predicted_video_dir: Optional[Path] = None,
) -> tuple[bool, list, list[np.ndarray]]:
    """Run a single episode and return success status, replay images, and predicted frames."""
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 30))

    env.reset()
    obs = env.set_init_state(initial_state)

    replay_images = []
    predicted_frames_list = []
    pending_actions: list[list[float]] = []
    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")

    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, pred_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                cfg=cfg,
                vlm_processor=vlm_processor,
                t5_encoder=t5_encoder,
                stat=stat,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
            predicted_frames_list.append(pred_frames)
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if done:
            break
        t += 1

    pbar.close()
    return bool(done), replay_images, predicted_frames_list


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    cfg: DictConfig,
    vlm_processor,
    t5_encoder,
    video_dir: Path,
    predicted_video_dir: Optional[Path],
    stat: Optional[dict] = None,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    """Run evaluation for a single task across multiple trials."""
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, pred_frames_list = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            cfg=cfg,
            vlm_processor=vlm_processor,
            t5_encoder=t5_encoder,
            episode_idx=trial_idx,
            stat=stat,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )

        if predicted_video_dir is not None and pred_frames_list:
            _save_predicted_frames(
                predicted_video_dir,
                pred_frames_list,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )

    return results


def _save_predicted_frames(
    pred_dir: Path,
    pred_frames_list: list[np.ndarray],
    idx: str,
    success: bool,
    task_description: str,
    fps: int = 8,
):
    """Save predicted future frames as MP4 video."""
    import imageio
    from PIL import Image, ImageDraw

    processed_desc = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]

    all_frames = []
    for frames in pred_frames_list:
        if frames.ndim == 4:
            if frames.shape[1] == 3:  # [T, C, H, W]
                frames = frames.transpose(0, 2, 3, 1)  # -> [T, H, W, C]
        for f in frames:
            f_uint8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            all_frames.append(f_uint8)

    if not all_frames:
        return

    mp4_path = pred_dir / f"{idx}--success={success}--task={processed_desc}--pred.mp4"
    writer = imageio.get_writer(str(mp4_path), fps=fps)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved predicted frames MP4: {mp4_path}")


def _create_ewam_model(cfg: DictConfig, device: str):
    """Create and load Ewam model from config and checkpoint."""
    from models.ewam import Ewam, EwamConfig

    common_cfg = cfg.common
    model_cfg = cfg.model

    ewam_config = EwamConfig(
        wan_checkpoint_path=model_cfg.wan.checkpoint_path,
        vae_path=model_cfg.wan.vae_path,
        wan_config_path=model_cfg.wan.config_path,
        video_precision=model_cfg.wan.precision,
        vlm_checkpoint_path=model_cfg.vlm.checkpoint_path,
        vlm_frozen=model_cfg.vlm.frozen,
        # Qwen3 expert
        vlm_dim=model_cfg.qwen3_expert.vlm_dim,
        qwen3_expert_head_dim=model_cfg.qwen3_expert.head_dim,
        qwen3_expert_num_heads=model_cfg.qwen3_expert.num_heads,
        qwen3_expert_num_layers=model_cfg.qwen3_expert.num_layers,
        qwen3_expert_norm_eps=model_cfg.qwen3_expert.norm_eps,
        # Action expert
        action_state_dim=common_cfg.state_dim,
        action_dim=common_cfg.action_dim,
        action_expert_dim=model_cfg.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=model_cfg.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=model_cfg.action_expert.norm_eps,
        # Video/sampling
        global_downsample_rate=common_cfg.global_downsample_rate,
        video_action_freq_ratio=common_cfg.video_action_freq_ratio,
        num_video_frames=common_cfg.num_video_frames,
        video_height=common_cfg.video_height,
        video_width=common_cfg.video_width,
        batch_size=1,
        training_mode=model_cfg.get('training_mode', 'finetune'),
        load_pretrained_backbones=False,
    )

    model = Ewam(ewam_config).to(device)
    model.load_checkpoint(str(cfg.ckpt), strict=False)
    model = model.eval()
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="ewam_libero_eval_horizontal_dim16")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()

    if cfg.get("seed") is not None:
        import random
        import numpy as np
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        torch.manual_seed(int(cfg.seed))

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    model_device = _resolve_eval_device(cfg)
    logging.info(f"Using device: {model_device}")

    # Create and load EWAM model
    logging.info("Loading EWAM model...")
    model = _create_ewam_model(cfg, model_device)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model.dtype = model_dtype  # Attach dtype for input conversion
    model = model.to(model_device).eval()
    logging.info(f"Model loaded successfully (action_dim={cfg.common.action_dim}, state_dim={cfg.common.state_dim})")

    # Initialize attention analysis (OPT-IN: only when ATTENTION_ANALYSIS_DIR is set
    # to a non-empty, non-"none" value). The default empty value keeps the original
    # evaluation behavior untouched — the parallel eval script doesn't set this var.
    # NOTE: This eval script runs as one process PER (suite, task_id) under the parallel
    # tmux scheduler. The save_dir MUST be unique per task to avoid parallel processes
    # overwriting each other's inference_step_*.pt files. We always append
    # {suite}/task{task_id}/ to the base dir.
    global _attention_capture
    if _ATTENTION_ANALYSIS_DIR and _ATTENTION_ANALYSIS_DIR != "none":
        from attention_analysis_mask import AttentionCapture
        _task_subdir = f"{cfg.EVALUATION.task_suite_name}/task{cfg.EVALUATION.task_id}"
        save_dir = f"{_ATTENTION_ANALYSIS_DIR}/{_task_subdir}"
        _attention_capture = AttentionCapture(save_dir, max_steps=_ATTENTION_MAX_STEPS)
        logging.info(f"Attention analysis enabled, save_dir={save_dir}")

        def _on_exit():
            if _attention_capture is not None:
                try:
                    _attention_capture.generate_report()
                except Exception as e:
                    logging.warning(f"Attention analysis report generation failed: {e}")
        atexit.register(_on_exit)

    # Load VLM processor
    vlm_processor = None
    vlm_ckpt = cfg.model.vlm.checkpoint_path
    if vlm_ckpt:
        try:
            from transformers import AutoProcessor
            vlm_processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=True)
            logging.info(f"VLM processor loaded from {vlm_ckpt}")
        except Exception as e:
            logging.warning(f"Failed to load VLM processor: {e}")

    # Load T5 encoder
    t5_encoder = None
    t5_config = cfg.get("t5", None)
    if t5_config is not None:
        try:
            from wan.modules.t5 import T5EncoderModel
            t5_encoder = T5EncoderModel(
                text_len=t5_config.get("text_len", 512),
                dtype=torch.bfloat16,
                device=str(model_device),
                checkpoint_path=t5_config.checkpoint_path,
                tokenizer_path=t5_config.tokenizer_path,
            )
            logging.info("T5 encoder loaded")
        except Exception as e:
            logging.warning(f"Failed to load T5 encoder: {e}")

    # Load normalization stats (aggregated across all suites, matching training)
    stat = None
    use_norm = bool(cfg.get("dataset", {}).get("use_normalization", False))
    if use_norm:
        try:
            stat = _load_normalization_stat(cfg)
            logging.info(f"Loaded normalization stats (state 8-D / action 7-D)")
            logging.info(f"  State min: {stat['observation.state']['min'][:3]}...")
            logging.info(f"  State max: {stat['observation.state']['max'][:3]}...")
            logging.info(f"  Action min: {stat['action']['min'][:3]}...")
            logging.info(f"  Action max: {stat['action']['max'][:3]}...")
        except Exception as e:
            logging.warning(f"Failed to load normalization stats: {e}")
            logging.warning("Falling back to no normalization")
    else:
        logging.info("use_normalization=false, skipping stat loading")

    # Determine action_horizon
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.common.num_video_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.common.get("video_size", [cfg.common.video_height, cfg.common.video_width])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    # Setup output directories
    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    predicted_video_dir.mkdir(parents=True, exist_ok=True)

    # Get task info
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with EWAM model (dim16)")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        cfg=cfg,
        vlm_processor=vlm_processor,
        t5_encoder=t5_encoder,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        stat=stat,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
