# LIBERO LeRobot Format Dataset Loader for EWAM
# Features (LIBERO style for state/action, RobotWin style for video/audio):
#   - State: EEF state (8D: position 3 + axis-angle 3 + gripper 2)
#   - Action: OSC delta (7D: position 3 + rotation 3 + gripper 1)
#   - Normalization: min/max normalization to [-1, 1]
#   - Delta mask: position/rotation are delta, gripper is absolute
#   - Sparse video sampling: sample video frames at video_action_freq_ratio intervals
#   - Video loading: use torchvision VideoReader (RobotWin style)
#   - Return format: RobotWin style (first_frame, video_frames, initial_state, action_sequence)

import os
import random
import json
import numpy as np
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings
import sys

# Import lerobot if available
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    logging.warning("LeRobot not installed, using fallback loader")

# Import image processing utilities (project root must be on PYTHONPATH — see launch scripts)
from pathlib import Path
import sys

PROJECT_ROOT = str((Path(__file__).parent.parent.parent).resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.utils.image_utils import tensor_to_pil, load_video_frames, get_video_frame_count

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Default task prefix following RobotWin style, adapted for LIBERO (single view + Franka)
DEFAULT_TASK_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: a front camera, a wrist camera and its copy. "
    "The Franka robot arm is currently performing the following task: "
)


class LiberoLeRobotDataset(data.Dataset):
    """
    Dataset for LIBERO using LeRobot format (min/max normalization + dim16 padding).

    Data structure:
    - videos/chunk-000/observation.images.image/episode_XXXXXX.mp4
    - videos/chunk-000/observation.images.wrist_image/episode_XXXXXX.mp4
    - data/chunk-000/episode_XXXXXX.parquet (contains state and action)
    - meta/tasks.jsonl (contains task instructions)
    - meta/stat.json (contains normalization statistics)

    LIBERO processing:
    - State: EEF state (8D: position 3 + axis-angle 3 + gripper 2)
    - Action: OSC delta (7D: position 3 + rotation 3 + gripper 1)
    - Delta mask: [true,true,true,true,true,true,false] - gripper is absolute
    - Normalization: min/max normalization

    Video processing (RobotWin style):
    - Video loading: torchvision VideoReader with pyav backend
    - Sparse sampling at video_action_freq_ratio intervals
    - T-shape concatenation of agentview + duplicated wrist

    Return format (RobotWin style):
    - first_frame: [C, H, W*2] - condition frame (agentview + wrist)
    - video_frames: [T, C, H, W*2] - sparse sampled video frames
    - initial_state: [8] - normalized EEF state at condition frame
    - action_sequence: [T, 7] - normalized actions
    - delta_action_mask: [7] - which dimensions are delta
    """

    # Delta mask: position/rotation (0-5) are delta, gripper (6) is absolute
    DELTA_ACTION_MASK = [True, True, True, True, True, True, False]

    def __init__(
        self,
        dataset_dirs: List[str],  # e.g., ["/path/to/libero_object_no_noops_lerobot"]
        suite_names: Union[str, List[str]] = "libero_object",

        # Sampling parameters
        global_downsample_rate: int = 3,
        video_action_freq_ratio: int = 2,
        num_video_frames: int = 8,

        # Video parameters
        video_size: Tuple[int, int] = (384, 320),

        # VLM processing
        vlm_checkpoint_path: Optional[str] = None,

        # State dimension (8 for EEF state)
        state_dim: int = 8,
        action_dim: int = 7,

        # Normalization (min/max to [-1, 1])
        stat_path: Optional[str] = None,  # Path to stat.json
        use_normalization: bool = True,

        # Additional params (for compatibility with training)
        max_episodes: Optional[int] = None,
        image_aug: bool = False,
        suite_weights: Optional[List[float]] = None,
        balance_suites: bool = False,
        task_name: Optional[str] = None,
        task_prefix: str = DEFAULT_TASK_PREFIX,  # Prefix for task instructions (like RobotWin)
        val: bool = False,
        image_concat_mode: str = "t_shape",  # "t_shape" or "horizontal"
        unify_to_16: bool = False,  # Pad action 7->16 / state 8->16 (UnifyTo16 layout) + emit action_dim_is_pad
    ):
        self.dataset_dirs = dataset_dirs if isinstance(dataset_dirs, list) else [dataset_dirs]
        self.suite_names = [suite_names] if isinstance(suite_names, str) else suite_names

        self.task_prefix = task_prefix  # Store task prefix for VLM

        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.video_size = video_size

        # Action chunk size: num_video_frames * video_action_freq_ratio
        self.action_chunk_size = num_video_frames * video_action_freq_ratio

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_normalization = use_normalization

        # Image concatenation mode: "t_shape" (RobotWin style) or "horizontal"
        if image_concat_mode not in ("t_shape", "horizontal"):
            raise ValueError(f"image_concat_mode must be 't_shape' or 'horizontal', got '{image_concat_mode}'")
        self.image_concat_mode = image_concat_mode

        # UnifyTo16: pad LIBERO 7-D action / 8-D state to 16-D (matching pretrain dims)
        # and emit action_dim_is_pad so the loss masks padded dims.
        self.unify_to_16 = unify_to_16
        # Action pad mask for 7-D LIBERO layout [eef(6), pad@6, grip@7, pad@8-15]:
        # True = padded dim (indices 6 and 8-15). State (8-D) pads indices 8-15 but is not a loss target.
        self._action_dim_is_pad = torch.tensor(
            [False] * 6 + [True] + [False] + [True] * 8, dtype=torch.bool
        )  # len 16

        # Delta mask
        self.delta_action_mask = torch.tensor(self.DELTA_ACTION_MASK, dtype=torch.bool)

        # Load normalization statistics
        self.stat = self._load_stats(stat_path)

        # VLM processor
        self.vlm_processor = None
        self.vlm_checkpoint_path = vlm_checkpoint_path
        if vlm_checkpoint_path:
            try:
                from transformers import AutoProcessor
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to load VLM processor: {e}")

        # Load episode index
        self.episodes = []
        self.tasks = {}  # task_index -> task string
        self._load_episodes()

        logger.info(f"LeRobot Dataset loaded: {len(self.episodes)} episodes from {len(self.dataset_dirs)} directories")
        logger.info(f"  num_video_frames: {num_video_frames}")
        logger.info(f"  video_action_freq_ratio: {video_action_freq_ratio}")
        logger.info(f"  action_chunk_size: {self.action_chunk_size}")
        logger.info(f"  use_normalization: {use_normalization}")
        logger.info(f"  image_concat_mode: {image_concat_mode}")
        logger.info(f"  unify_to_16: {unify_to_16}")
        logger.info(f"  delta_action_mask: {self.DELTA_ACTION_MASK}")

    def _load_stats(self, stat_path: Optional[str] = None) -> Dict:
        """Load normalization statistics, aggregating across ALL dataset_dirs .

        Aggregates global_min/global_max across all episodes/suites
        (element-wise min of per-episode mins, max of per-episode maxs). We replicate by reading
        each dataset_dir's stat file and taking element-wise min/max across suites — so a model
        trained on all 4 LIBERO suites uses the union range, not just the first suite's range.

        Supports two per-file formats:
        1. Format: {state: {default: {global_min, global_max, ...}}, action: {default: {...}}}
        2. Flat format: {observation.state: {min, max}, action: {min, max}}

        If stat_path is explicitly provided, uses that single file (no aggregation) — used by eval
        to load a training run's pre-saved dataset_stats.json.
        Otherwise aggregates across all dataset_dirs.

        Returns flat format: {observation.state: {min, max}, action: {min, max}}
        """
        def _extract_flat(raw: Dict) -> Dict:
            """Extract flat {observation.state: {min, max}, action: {min, max}} from one stat dict."""
            flat = {}
            # Format: {state: {default: {global_min, global_max}}, action: {default: {...}}}
            if "state" in raw and isinstance(raw["state"], dict) and "default" in raw["state"]:
                for src_key, dst_key in [("state", "observation.state"), ("action", "action")]:
                    if src_key in raw and "default" in raw[src_key]:
                        entry = raw[src_key]["default"]
                        mins = entry.get("global_min", entry.get("min"))
                        maxs = entry.get("global_max", entry.get("max"))
                        if mins is not None and maxs is not None:
                            flat[dst_key] = {"min": list(mins), "max": list(maxs)}
                return flat
            # Flat format: {observation.state: {min, max}, action: {min, max}}
            for src_key, dst_key in [("observation.state", "observation.state"),
                                     ("state", "observation.state"),
                                     ("action", "action")]:
                if (src_key in raw and isinstance(raw[src_key], dict)
                        and "min" in raw[src_key] and "max" in raw[src_key]
                        and dst_key not in flat):
                    flat[dst_key] = {"min": list(raw[src_key]["min"]), "max": list(raw[src_key]["max"])}
            return flat

        # Explicit single-file path: use it directly (no aggregation) — eval path.
        if stat_path and Path(stat_path).exists():
            with open(stat_path, 'r') as f:
                raw = json.load(f)
            logger.info(f"Loaded stats (single file, no aggregation) from {stat_path}")
            flat = _extract_flat(raw)
            return flat if flat else raw

        # Aggregate across ALL dataset_dirs (union range across suites).
        collected = {}  # key -> list of (mins, maxs) per suite
        for dataset_dir in self.dataset_dirs:
            for candidate_name in ["meta/stat.json", "dataset_stats.json"]:
                candidate = Path(dataset_dir) / candidate_name
                if not candidate.exists():
                    continue
                with open(candidate, 'r') as f:
                    raw = json.load(f)
                flat = _extract_flat(raw)
                for key, entry in flat.items():
                    collected.setdefault(key, []).append((entry["min"], entry["max"]))
                logger.info(f"  stat source: {candidate}")
                break  # one stat file per dataset_dir (prefer meta/stat.json)

        if collected:
            aggregated = {}
            for key, entries in collected.items():
                mins_arr = [e[0] for e in entries]
                maxs_arr = [e[1] for e in entries]
                # element-wise min of mins, max of maxs across suites
                agg_min = [min(col) for col in zip(*mins_arr)]
                agg_max = [max(col) for col in zip(*maxs_arr)]
                aggregated[key] = {"min": agg_min, "max": agg_max}
            n_suites = len(next(iter(collected.values())))
            logger.info(f"Aggregated normalization stats across {n_suites} suite(s) (element-wise min/max):")
            for key in aggregated:
                logger.info(f"  {key}: min={[round(v, 4) for v in aggregated[key]['min']]}")
                logger.info(f"  {key}: max={[round(v, 4) for v in aggregated[key]['max']]}")
            return aggregated

        # Default statistics
        logger.warning("No stat.json found, using default normalization range [-1, 1]")
        return {
            "observation.state": {
                "min": [-1] * 8,
                "max": [1] * 8,
            },
            "action": {
                "min": [-1] * 7,
                "max": [1] * 7,
            }
        }

    def _normalize(self, values: torch.Tensor, key: str) -> torch.Tensor:
        """Normalize values to [-1, 1] range using min/max normalization.

        Linear normalizer with mode="min/max":
            scale = (output_max - output_min) / (input_max - input_min)
            offset = output_min - scale * input_min
            normalized = values * scale + offset
            clamp to [-5, 5]
        """
        if not self.use_normalization:
            return values

        dim = values.shape[-1]
        key_name = key

        if key_name in self.stat and "min" in self.stat[key_name] and "max" in self.stat[key_name]:
            mins = torch.tensor(self.stat[key_name]["min"][-dim:], device=values.device, dtype=values.dtype)
            maxs = torch.tensor(self.stat[key_name]["max"][-dim:], device=values.device, dtype=values.dtype)
        else:
            mins = torch.full((dim,), -1.0, device=values.device, dtype=values.dtype)
            maxs = torch.full((dim,), 1.0, device=values.device, dtype=values.dtype)

        output_min, output_max = -1.0, 1.0
        ranges = maxs - mins
        # For dimensions where range is near-zero, map to center of output range
        ignore_dim = ranges < 1e-4
        ranges = torch.where(ignore_dim, torch.tensor(output_max - output_min, dtype=values.dtype, device=values.device), ranges)

        scale = (output_max - output_min) / ranges
        offset = output_min - scale * mins
        # Near-zero range dims: center at (output_max + output_min) / 2
        offset = torch.where(ignore_dim, (output_max + output_min) / 2.0 - mins, offset)

        normalized = values * scale + offset
        normalized = torch.clamp(normalized, -5.0, 5.0)
        return normalized

    @staticmethod
    def _unify_action_to_16(action: torch.Tensor) -> torch.Tensor:
        """Pad 7-D LIBERO action to 16-D UnifyTo16 layout: [eef(6), pad@6, grip@7, pad@8-15].

        UnifyTo16 (dim==7 branch) layout. Pad dims are 0 in normalized space.
        """
        assert action.shape[-1] == 7, f"expected 7-D action, got {action.shape[-1]}"
        eef = action[..., :6]
        grip = action[..., 6:7]
        pad1 = torch.zeros_like(grip)  # index 6
        right_pad = torch.zeros(*action.shape[:-1], 8, dtype=action.dtype, device=action.device)  # indices 8-15
        return torch.cat([eef, pad1, grip, right_pad], dim=-1)  # 6+1+1+8 = 16

    @staticmethod
    def _unify_state_to_16(state: torch.Tensor) -> torch.Tensor:
        """Pad 8-D LIBERO state to 16-D UnifyTo16 layout: [eef(6), grip_open@6, grip_close@7, pad@8-15].

        UnifyTo16 (dim==8 branch) layout. State is conditioning (not a loss target), so no mask needed.
        """
        assert state.shape[-1] == 8, f"expected 8-D state, got {state.shape[-1]}"
        left = state[..., :7]  # eef(6) + grip_open(1)
        grip = state[..., 7:8]  # grip_close
        right_pad = torch.zeros(*state.shape[:-1], 8, dtype=state.dtype, device=state.device)  # indices 8-15
        return torch.cat([left, grip, right_pad], dim=-1)  # 7+1+8 = 16

    def _load_tasks(self, dataset_path: Path):
        """Load task instructions from tasks.jsonl."""
        tasks_file = dataset_path / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file, 'r') as f:
                for line in f:
                    task_data = json.loads(line.strip())
                    self.tasks[task_data['task_index']] = task_data['task']
            logger.info(f"Loaded {len(self.tasks)} tasks from {tasks_file}")

    def _load_episodes(self):
        """Load all episodes from LeRobot format directories."""
        for dataset_dir in self.dataset_dirs:
            dataset_path = Path(dataset_dir)

            # Load task instructions
            self._load_tasks(dataset_path)

            # Load episodes from parquet files
            data_dir = dataset_path / "data" / "chunk-000"
            video_dir = dataset_path / "videos" / "chunk-000"

            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}")
                continue

            parquet_files = sorted(data_dir.glob("episode_*.parquet"))
            for pq_file in parquet_files:
                ep_name = pq_file.stem  # episode_XXXXXX

                # Video paths
                image_video = video_dir / "observation.images.image" / f"{ep_name}.mp4"
                wrist_video = video_dir / "observation.images.wrist_image" / f"{ep_name}.mp4"

                if not image_video.exists():
                    continue

                # Get task_index from parquet
                import pandas as pd
                df = pd.read_parquet(pq_file)
                task_index = int(df['task_index'].iloc[0]) if 'task_index' in df.columns else 0
                task = self.tasks.get(task_index, "")

                self.episodes.append({
                    'episode_name': ep_name,
                    'parquet_path': str(pq_file),
                    'image_video_path': str(image_video),
                    'wrist_video_path': str(wrist_video),
                    'dataset_dir': str(dataset_path),
                    'task_index': task_index,
                    'task': task,
                    't5_embedding_path': str(dataset_path / "t5_embedding" / f"{ep_name}.pt"),
                })

    def _load_sample_from_parquet(self, parquet_path: str) -> Dict:
        """Load state and action from parquet file."""
        import pandas as pd

        df = pd.read_parquet(parquet_path)

        # Get state and action arrays
        states = np.stack(df['observation.state'].values)  # (T, 8)
        actions = np.stack(df['action'].values)  # (T, 7)

        return {
            'states': torch.from_numpy(states).float(),
            'actions': torch.from_numpy(actions).float(),
        }

    def _load_t5_embedding(self, embed_path: str) -> Optional[torch.Tensor]:
        """Load T5 embedding from .pt file.

        Returns:
            Tensor of shape [S, D] (seq_len, hidden_dim) on CPU, or None if not found.
        """
        if embed_path is None or not Path(embed_path).exists():
            return None

        try:
            embed = torch.load(embed_path, map_location='cpu')
            # Handle [1, S, D] -> [S, D]
            if embed.dim() == 3 and embed.shape[0] == 1:
                embed = embed.squeeze(0)
            # Ensure it's float32 for consistency
            if embed.dtype != torch.float32:
                embed = embed.float()
            return embed
        except Exception as e:
            logger.warning(f"Failed to load T5 embedding from {embed_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.episodes) * 10  # Each episode is sampled 10 times per epoch (same as RoboTwin)

    def _load_video_torchvision(self, video_path: str, indices: List[int], target_size: Optional[Tuple[int, int]]) -> torch.Tensor:
        """Load video frames using torchvision VideoReader (RobotWin style)."""
        import torchvision
        torchvision.set_video_backend('pyav')
        from torchvision.io import VideoReader

        reader = VideoReader(video_path)
        fps = 20.0

        frames = []
        for idx in indices:
            timestamp = idx / fps
            reader.seek(timestamp, keyframes_only=True)

            for frame in reader:
                frames.append(frame['data'])
                break

        reader = None

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_path}")

        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.float() / 255.0

        if target_size is not None and frames.shape[-2:] != target_size:
            frames = torch.nn.functional.interpolate(
                frames, size=target_size, mode='bilinear', align_corners=False
            )

        return frames

    def _concat_t_shape(self, agent_frames: torch.Tensor, wrist_frames: torch.Tensor) -> torch.Tensor:
        """Concatenate agentview and wrist camera in T-shape layout, then resize to self.video_size.

        Layout:
          ┌──────────────┐
          │  agentview   │  H x W
          ├──────┬───────┤
          │ wrist│ wrist │  H/2 x W (each wrist is H/2 x W/2, duplicated horizontally)
          └──────┴───────┘
          Raw result: 3H/2 x W, then resize to self.video_size

        Args:
            agent_frames: [T, C, H, W] agentview frames in [0,1]
            wrist_frames: [T, C, H, W] wrist frames in [0,1]

        Returns:
            [T, C, video_H, video_W] t-shape concatenated and resized frames
        """
        T, C, H, W = agent_frames.shape
        half_h, half_w = H // 2, W // 2

        # Resize wrist to half size
        wrist_resized = torch.nn.functional.interpolate(
            wrist_frames, size=(half_h, half_w), mode='bilinear', align_corners=False
        )  # [T, C, half_h, half_w]

        # Duplicate wrist horizontally: [T, C, half_h, W]
        wrist_dup = torch.cat([wrist_resized, wrist_resized], dim=3)

        # Stack vertically: [T, C, H + half_h, W]
        combined = torch.cat([agent_frames, wrist_dup], dim=2)

        # Resize to target video_size
        if self.video_size is not None and combined.shape[-2:] != self.video_size:
            combined = torch.nn.functional.interpolate(
                combined, size=self.video_size, mode='bilinear', align_corners=False
            )

        return combined

    def _concat_horizontal(self, agent_frames: torch.Tensor, wrist_frames: torch.Tensor) -> torch.Tensor:
        """Concatenate agentview and wrist camera horizontally.

        Each camera is resized to full [H, W] = video_size before concatenation.
        The result [H, 2W] is then resized to video_size [H, W] .
        Horizontal concat:
          1. Resize each camera to shape_meta.shape = [H, W]
          2. Concat horizontally: [H, W] + [H, W] = [H, 2W]
          3. Final resize_transform to video_size = [H, W]

        Layout:
          ┌────────────┬────────────┐
          │  agentview │   wrist    │  H x (W + W) = H x 2W
          └────────────┴────────────┘
          Final: resize to video_size = self.video_size

        Args:
            agent_frames: [T, C, H, W] agentview frames in [0,1]
            wrist_frames: [T, C, H, W] wrist frames in [0,1]

        Returns:
            [T, C, video_H, video_W] horizontally concatenated and resized frames
        """
        # First resize each camera to full [H, W] = video_size (per-camera resize)
        if self.video_size is not None:
            target_h, target_w = self.video_size
        else:
            _, _, H, W = agent_frames.shape
            target_h, target_w = H, W

        agent_resized = torch.nn.functional.interpolate(
            agent_frames, size=(target_h, target_w), mode='bilinear', align_corners=False
        )  # [T, C, H, W]
        wrist_resized = torch.nn.functional.interpolate(
            wrist_frames, size=(target_h, target_w), mode='bilinear', align_corners=False
        )  # [T, C, H, W]

        # Concatenate horizontally: [T, C, H, W + W] = [T, C, H, 2W]
        combined = torch.cat([agent_resized, wrist_resized], dim=3)

        # Final resize to video_size 
        if self.video_size is not None:
            combined = torch.nn.functional.interpolate(
                combined, size=self.video_size, mode='bilinear', align_corners=False
            )

        return combined

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        """
        Calculate sampling indices following RobotWin style with random start.

        Key principle: len(action_indices) = num_video_frames × video_action_freq_ratio
        video_indices are sampled from action_indices at action_step = (i+1)*ratio - 1
        condition_frame is a random position within the episode (same as RoboTwin)

        Returns:
            cond_idx: condition frame index (random start within episode)
            video_indices: indices for video frames to predict
            action_indices: full action indices (action_chunk_size entries)
        """
        # Calculate action_chunk_size (same as RoboTwin)
        action_chunk_size = self.num_video_frames * self.video_action_freq_ratio

        # Calculate physical span of one chunk
        physical_chunk_size = action_chunk_size * self.global_downsample_rate

        # Random condition frame (same as RoboTwin)
        max_condition_idx = total_frames - physical_chunk_size - 1
        if max_condition_idx < 0:
            cond_idx = 0
        else:
            cond_idx = random.randint(0, max_condition_idx)

        # Generate action_indices: action_chunk_size entries
        # Starting at cond_idx+1, one action every global_downsample_rate frames
        action_indices = []
        for i in range(action_chunk_size):
            action_idx = cond_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))

        # Sample video_indices from action_indices at action_step intervals
        # video_indices[i] = action_indices[(i+1)*ratio - 1]
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])

        return cond_idx, video_indices, action_indices

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a training sample (RobotWin style return format)."""
        max_attempts = 5

        for _ in range(max_attempts):
            try:
                # Random episode selection (same as RoboTwin)
                episode = random.choice(self.episodes)

                # Load all parquet data
                parquet_data = self._load_sample_from_parquet(episode['parquet_path'])
                states = parquet_data['states']  # (T, 8)
                actions = parquet_data['actions']  # (T, 7)

                total_frames = len(states)
                if total_frames < self.num_video_frames + 1:
                    idx = random.randint(0, len(self) - 1)
                    continue

                # Calculate sparse indices
                cond_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)

                # Load videos
                first_frame = self._load_video_torchvision(episode['image_video_path'], [cond_idx], None)
                first_wrist = self._load_video_torchvision(episode['wrist_video_path'], [cond_idx], None)
                video_frames = self._load_video_torchvision(episode['image_video_path'], video_indices, None)
                wrist_frames = self._load_video_torchvision(episode['wrist_video_path'], video_indices, None)

                # T-shape concatenation: agentview top + duplicated wrist bottom
                # Layout:
                #   ┌──────────┐
                #   │ agentview│  H x W
                #   ├───┬──────┤
                #   │wst│ wst  │  H/2 x W (each wrist is H/2 x W/2, duplicated)
                #   └───┴──────┘
                #   Final: 3H/2 x W, then resize to self.video_size
                if self.image_concat_mode == "t_shape":
                    first_frame = self._concat_t_shape(first_frame, first_wrist)
                    video_frames = self._concat_t_shape(video_frames, wrist_frames)
                else:  # "horizontal"
                    first_frame = self._concat_horizontal(first_frame, first_wrist)
                    video_frames = self._concat_horizontal(video_frames, wrist_frames)

                # Get state and action at indices
                initial_state = states[cond_idx]  # (8,) EEF state
                action_sequence = actions[action_indices]  # (T, 7) OSC delta

                # Normalize state and action
                initial_state_normalized = self._normalize(initial_state.unsqueeze(0), "observation.state").squeeze(0)
                action_sequence_normalized = self._normalize(action_sequence, "action")

                # UnifyTo16: pad action 7->16 and state 8->16 (AFTER normalization, so
                # padded dims are 0 in normalized space). Emit per-dim pad mask for the loss.
                action_dim_is_pad = None
                if self.unify_to_16:
                    action_sequence_normalized = self._unify_action_to_16(action_sequence_normalized)
                    initial_state_normalized = self._unify_state_to_16(initial_state_normalized)
                    action_dim_is_pad = self._action_dim_is_pad  # [16] bool, True=pad

                # Get task instruction with prefix (for consistency with T5 embedding)
                raw_task = episode.get('task', '')
                if not raw_task:
                    raw_task = self.tasks.get(episode.get('task_index', 0), '')
                task = self.task_prefix + raw_task

                # VLM processing (RobotWin style)
                vlm_inputs = None
                if self.vlm_processor is not None and task:
                    from utils.vlm_utils import preprocess_vlm_messages
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    vlm_inputs = preprocess_vlm_messages(task, first_frame_pil, self.vlm_processor)

                # Load T5 language embedding (episode-level)
                language_embedding = self._load_t5_embedding(episode.get('t5_embedding_path'))

                return {
                    'first_frame': first_frame.squeeze(0),
                    'video_frames': video_frames,
                    'initial_state': initial_state_normalized,  # Normalized (16-D if unify_to_16)
                    'action_sequence': action_sequence_normalized,  # Normalized (16-D if unify_to_16)
                    'initial_state_raw': initial_state,  # Raw for reference (8-D)
                    'action_sequence_raw': action_sequence,  # Raw for reference (7-D)
                    'action_dim_is_pad': action_dim_is_pad,  # [16] bool (True=pad) or None
                    'delta_action_mask': self.delta_action_mask,
                    'language_embedding': language_embedding,
                    'vlm_inputs': vlm_inputs,
                    'task': task,
                    'suite_name': episode['dataset_dir'].split('/')[-1],
                    'task_name': episode['episode_name'],
                }

            except Exception as e:
                import traceback
                logger.warning(f"Sample error: {e}\n{traceback.format_exc()}")
                idx = random.randint(0, len(self) - 1)
                continue

        return None


if __name__ == "__main__":
    # Quick test of the dataset
    import os
    from pathlib import Path
    from data.dataset import create_dataset
    from omegaconf import OmegaConf

    # Get absolute path to config
    script_dir = Path(__file__).parent
    config_path = script_dir.parent.parent / "configs" / "libero_lerobot.yaml"

    config = OmegaConf.load(str(config_path))
    config.model.vlm.checkpoint_path = None
    config.model.wan.checkpoint_path = None

    dataset = create_dataset(config, val=False)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    if sample:
        print(f"First frame shape: {sample['first_frame'].shape}")
        print(f"Video frames shape: {sample['video_frames'].shape}")
        print(f"Initial state shape: {sample['initial_state'].shape}")
        print(f"Action sequence shape: {sample['action_sequence'].shape}")
        print(f"Delta action mask: {sample['delta_action_mask']}")
        print(f"\\nNormalized state range: [{sample['initial_state'].min():.3f}, {sample['initial_state'].max():.3f}]")
        print(f"Normalized action range: [{sample['action_sequence'].min():.3f}, {sample['action_sequence'].max():.3f}]")
        print(f"\\nTask: {sample.get('task', '')}")
    else:
        print("ERROR: Failed to load sample")
