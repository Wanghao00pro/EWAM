# EWAM Policy for RoboTwin
#
# Deployment policy implementing the RoboTwin benchmark policy interface:
#   encode_obs(observation) / get_model(usr_args) / eval(TASK_ENV, model, observation) / reset_model(model)
#
# Model preprocessing must match RoboTwin dim16 training exactly:
#   - Image: T-shape layout (head camera on top, left|right wrist cameras side-by-side below),
#     resized with padding to (video_height, video_width) from the training config.
#   - Instruction: the same scene prefix that the RoboTwin data converter prepends to every
#     training instruction (see data/robotwin2/robotwin_data_convert/robotwin_converter.py).
#   - State: dual-arm qpos [14] -> padded to the unified 16-dim layout
#     [left_joints(6), pad(1), left_gripper(1), right_joints(6), pad(1), right_gripper(1)].
#     RoboTwin training uses RAW (un-normalized) state/actions, so no normalization here.
#   - Action: the model's 16-dim output chunk is restored to 14-dim qpos (valid dims
#     [0..5, 7, 8..13, 15]) before being sent to TASK_ENV.take_action(action_type='qpos').
#
# Required usr_args / env vars (see README "RoboTwin → Evaluation"):
#   ckpt_setting : path to the trained checkpoint (checkpoint_step_N dir or
#                  mp_rank_00_model_states.pt file)
#   wan_path     : Wan2.2-TI2V-5B directory (T5 encoder + VAE + wan config)
#   vlm_path     : Qwen3-VL-2B-Instruct directory (processor; weights come from the checkpoint)
#   config_path  : EWAM training config yaml used for the checkpoint (architecture only)

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import cv2
from PIL import Image
from transformers import AutoProcessor
from typing import List, Dict, Any, Optional
from collections import deque
import logging

# ---------------------------------------------------------------------------
# Path setup: locate the EWAM repository root so the package modules can be
# imported. Resolution order: EWAM_ROOT env var -> this file's location inside
# the repo (EWAM/inference/robotwin/deploy_policy.py).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent


def _find_ewam_root() -> Optional[str]:
    env_root = os.environ.get("EWAM_ROOT")
    if env_root and (Path(env_root) / "models").is_dir():
        return str(Path(env_root).resolve())
    probe = _HERE
    for _ in range(5):
        if (probe / "models" / "ewam.py").exists():
            return str(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


EWAM_ROOT = _find_ewam_root()
if EWAM_ROOT is not None and EWAM_ROOT not in sys.path:
    sys.path.insert(0, EWAM_ROOT)

from models.ewam import Ewam, EwamConfig  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # noqa: E402
from data.utils.image_utils import resize_with_padding, tensor_to_pil  # noqa: E402
from utils.vlm_utils import preprocess_vlm_messages  # noqa: E402

logger = logging.getLogger(__name__)

# Scene prefix — MUST match the converter's meta_prefix used at training-data
# generation time (robotwin_data_convert/robotwin_converter.py).
SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def _unify_qpos_to_16(x: np.ndarray) -> np.ndarray:
    """Pad [.., 14] dual-arm qpos to the unified 16-dim layout (mirror of the
    training dataset's _unify_action_to_16 for dim=14):
    [left_joints(6), pad(1), left_gripper(1), right_joints(6), pad(1), right_gripper(1)]
    """
    if x.shape[-1] == 16:
        return x
    assert x.shape[-1] == 14, f"Expected 14-dim qpos, got {x.shape[-1]}-d"
    left_joints = x[..., :6]
    g0 = x[..., 6:7]
    right_joints = x[..., 7:13]
    g1 = x[..., 13:14]
    pad = np.zeros((*x.shape[:-1], 1), dtype=x.dtype)
    return np.concatenate([left_joints, pad, g0, right_joints, pad, g1], axis=-1)


def _restore_action_to_14(action: np.ndarray) -> np.ndarray:
    """Restore a 16-dim unified action chunk to 14-dim qpos: keep dims [0..5, 7, 8..13, 15]."""
    assert action.shape[-1] == 16, f"Expected 16-dim action, got {action.shape[-1]}-d"
    return action[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15]]


class EwamPolicy:
    """EWAM Policy wrapper for RoboTwin evaluation (joint video-action flow matching)."""

    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str,
                 device: str = "cuda", log_dir: Optional[str] = None,
                 task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path

        # Load the training config (architecture + video layout + inference settings)
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        # Initialize model WITHOUT pretrained backbones, then load the full checkpoint
        self.model = self._load_model()

        # T5 encoder for the video model's cross-attention conditioning
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.wan_path, "google", "umt5-xxl"),
        )

        # VLM processor (tokenization only; VLM weights come from the checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)

        # Observation / action caches
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        self.current_state = None
        self.current_instruction = ""

        # Image saving (debug visualization)
        self.save_images = True
        base_log_dir = log_dir or os.environ.get("LOG_DIR") or str(_HERE / "logs")
        task_dir_name = task_name or os.environ.get("TASK_NAME") or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("EWAM Policy initialized successfully")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _load_model(self) -> Ewam:
        logger.info("Initializing Ewam from config (no pretrained backbones)")
        config = self._create_model_config()
        model = Ewam(config)
        model = model.to(self.device)

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        model.load_checkpoint(self.checkpoint_path, strict=False)
        logger.info("Model checkpoint loaded successfully")

        model.dtype = torch.bfloat16  # attach dtype for input conversion (see eval scripts)
        model.eval()
        return model

    def _create_model_config(self) -> EwamConfig:
        """Create model configuration from the training yaml — inference mode."""
        common = self.config_dict["common"]
        model_cfg = self.config_dict["model"]

        config = EwamConfig(
            wan_checkpoint_path=self.wan_path,
            vae_path=os.path.join(self.wan_path, "Wan2.2_VAE.pth"),
            wan_config_path=self.wan_path,
            video_precision=model_cfg["wan"].get("precision", "bfloat16"),
            vlm_checkpoint_path=self.vlm_path,
            vlm_frozen=model_cfg["vlm"].get("frozen", False),
            # Qwen3 expert (per-layer QKV fusion)
            vlm_dim=model_cfg["qwen3_expert"]["vlm_dim"],
            qwen3_expert_head_dim=model_cfg["qwen3_expert"]["head_dim"],
            qwen3_expert_num_heads=model_cfg["qwen3_expert"]["num_heads"],
            qwen3_expert_num_layers=model_cfg["qwen3_expert"]["num_layers"],
            qwen3_expert_norm_eps=model_cfg["qwen3_expert"]["norm_eps"],
            # Action expert
            action_state_dim=common["state_dim"],
            action_dim=common["action_dim"],
            action_expert_dim=model_cfg["action_expert"]["hidden_size"],
            action_expert_ffn_dim_multiplier=model_cfg["action_expert"]["ffn_dim_multiplier"],
            action_expert_norm_eps=model_cfg["action_expert"]["norm_eps"],
            # Video / sampling
            global_downsample_rate=common["global_downsample_rate"],
            video_action_freq_ratio=common["video_action_freq_ratio"],
            num_video_frames=common["num_video_frames"],
            video_height=common["video_height"],
            video_width=common["video_width"],
            batch_size=1,
            training_mode=model_cfg.get("training_mode", "finetune"),
            load_pretrained_backbones=False,
        )
        return config

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def set_instruction(self, instruction: str):
        """Set the current (raw) task instruction. The scene prefix is added in get_action."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def update_obs(self, observation: Dict[str, Any]):
        """Update the observation cache (T-shape image + 16-dim padded state)."""
        # --- Image: T-shape (head top; left|right wrist bottom) ---
        if "observation" in observation:
            obs_data = observation["observation"]
            if "head_camera" in obs_data and "left_camera" in obs_data and "right_camera" in obs_data:
                head_img = obs_data["head_camera"]["rgb"]
                left_img = obs_data["left_camera"]["rgb"]
                right_img = obs_data["right_camera"]["rgb"]

                left_img_resized = cv2.resize(left_img, (left_img.shape[1] // 2, left_img.shape[0] // 2))
                right_img_resized = cv2.resize(right_img, (right_img.shape[1] // 2, right_img.shape[0] // 2))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data (need head_camera / left_camera / right_camera)")
        elif "head_camera" in observation:
            image = observation["head_camera"]
        elif "image" in observation:
            image = observation["image"]
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict["common"]["video_height"],
                       self.config_dict["common"]["video_width"])

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        elif image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        self.obs_cache.append(image_tensor.to(self.device))

        # --- State: dual-arm qpos [14] -> padded 16-dim, RAW (no normalization) ---
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        state = _unify_qpos_to_16(state)
        self.current_state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def get_action(self, instruction: Optional[str] = None) -> List[np.ndarray]:
        """Run one inference step and return the action chunk as a list of 14-dim qpos arrays."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        if instruction is not None:
            self.set_instruction(instruction)

        current_frame = self.obs_cache[-1]

        # Prefixed instruction (matches training metas)
        prefixed_instruction = f"{SCENE_PREFIX}{self.current_instruction}"

        # T5 language embeddings for the video model's cross-attention
        t5_out = self.t5_encoder([prefixed_instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # VLM inputs for the understanding stream
        first_frame_pil = tensor_to_pil(current_frame.squeeze(0).cpu())
        vlm_inputs = preprocess_vlm_messages(prefixed_instruction, first_frame_pil, self.vlm_processor)
        if vlm_inputs is not None:
            vlm_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in vlm_inputs.items()}

        num_inference_steps = self.config_dict["model"]["inference"]["num_inference_timesteps"]
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs] if vlm_inputs else None,
            )

        # Debug visualization (condition | 4 predicted frames)
        if self.save_images and predicted_frames is not None and predicted_frames.dim() == 5:
            try:
                frames_viz = predicted_frames.squeeze(0)          # [T, C, H, W]
                if frames_viz.shape[0] == 3:                      # [C, T, H, W] fallback
                    frames_viz = frames_viz.permute(1, 0, 2, 3)
                self._save_frame_grid(current_frame.squeeze(0), frames_viz)
                self.step_count += 1
            except Exception as e:
                logger.warning(f"Failed to save frame grid: {e}")

        # predicted_actions: [B, chunk, 16] raw qpos space (RoboTwin training is un-normalized)
        actions_16 = predicted_actions.squeeze(0).cpu().float().numpy()   # [chunk, 16]
        actions_14 = _restore_action_to_14(actions_16)                    # [chunk, 14]
        self.action_cache.extend(list(actions_14))
        return list(actions_14)

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        def tensor_to_numpy(t):
            if t.dim() == 3:
                t = t.permute(1, 2, 0)
            t = t.detach().cpu().float().clamp(0, 1)
            return (t.numpy() * 255).astype(np.uint8)

        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = [tensor_to_numpy(predicted_frames[i]) for i in range(predicted_frames.shape[0])]
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        grid_image = np.concatenate([condition_np] + predicted_np[:4], axis=1)
        filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
        grid_image = Image.fromarray(grid_image)
        grid_image.save(self.save_dir / filename)


# ---------------------------------------------------------------------------
# RoboTwin benchmark policy interface
# ---------------------------------------------------------------------------
def encode_obs(observation):
    """Post-process observation (identity — all processing happens in update_obs)."""
    return observation


def get_model(usr_args):
    """Initialize the EWAM policy from RoboTwin's eval usr_args."""
    checkpoint_path = usr_args.get("ckpt_setting")
    wan_path = usr_args.get("wan_path") or os.environ.get("EWAM_WAN_PATH")
    vlm_path = usr_args.get("vlm_path") or os.environ.get("EWAM_VLM_PATH")
    config_path = usr_args.get("config_path") or os.environ.get("EWAM_CONFIG")

    if not checkpoint_path:
        raise ValueError("ckpt_setting not provided in usr_args")
    if not wan_path:
        raise ValueError("wan_path not provided (usr_args or EWAM_WAN_PATH env var)")
    if not vlm_path:
        raise ValueError("vlm_path not provided (usr_args or EWAM_VLM_PATH env var)")
    if not config_path:
        # Default to the RoboTwin stage-2 training config shipped with EWAM
        if EWAM_ROOT is None:
            raise RuntimeError("EWAM repository root not found; set the EWAM_ROOT env var")
        config_path = os.path.join(EWAM_ROOT, "configs", "ewam_robotwin.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return EwamPolicy(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        device=device,
        log_dir=usr_args.get("log_dir"),
        task_name=usr_args.get("task_name"),
    )


def eval(TASK_ENV, model, observation):
    """Evaluation step: one observation -> one action chunk -> execute."""
    obs = encode_obs(observation)

    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model):
    """Reset per-episode state."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
