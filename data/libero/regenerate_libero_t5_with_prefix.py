#!/usr/bin/env python3
"""
Regenerate T5 embeddings for LIBERO LeRobot dataset with a prompt prefix.

Usage:
    cd <EWAM root>
    python data/libero/regenerate_libero_t5_with_prefix.py \
        --dataset_dir /path/to/libero_object_no_noops_lerobot \
        --wan_path /path/to/Wan2.2-TI2V-5B

Prefix for LIBERO (must match LiberoLeRobotDataset.DEFAULT_TASK_PREFIX):
    "The whole scene is in a realistic, industrial art style with three views: a front camera,
     a wrist camera and its copy. The Franka robot is currently performing the following task: "
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Default prefix for LIBERO (three views: front camera, wrist camera, and its copy)
DEFAULT_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: a front camera, a wrist camera and its copy. "
    "The Franka robot arm is currently performing the following task: "
)


def load_t5_encoder(wan_path: str, device: str):
    """Initialize WAN T5EncoderModel."""
    # The `wan` package (Wan2.2 official code) lives at the EWAM project root
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from wan.modules.t5 import T5EncoderModel

    ckpt = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "google/umt5-xxl")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"T5 checkpoint not found: {ckpt}")
    if not os.path.exists(tok):
        raise FileNotFoundError(f"T5 tokenizer dir not found: {tok}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def encode_t5(encoder, instruction: str, device: str) -> torch.Tensor:
    """Encode instruction to T5 embedding."""
    with torch.no_grad():
        out = encoder([instruction], device)
    if isinstance(out, list):
        emb = out[0]
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        raise ValueError(f"Unexpected T5 encoder output type: {type(out)}")

    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    return emb.detach().cpu()


def regenerate_t5_embeddings(
    dataset_dir: str,
    wan_path: str,
    prefix: str,
    device: str = "cuda",
    overwrite: bool = True,
):
    """
    Regenerate T5 embeddings with prompt prefix for LIBERO dataset.
    """
    dataset_path = Path(dataset_dir)

    # Load tasks from tasks.jsonl
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_file}")

    tasks = {}
    with open(tasks_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            tasks[data['task_index']] = data['task']

    print(f"Loaded {len(tasks)} tasks")

    # Create t5_embedding directory
    t5_dir = dataset_path / "t5_embedding"
    t5_dir.mkdir(parents=True, exist_ok=True)

    # Build prefixed tasks
    tasks_with_prefix = {}
    for task_index, task in tasks.items():
        prefixed_task = f"{prefix}{task}"
        tasks_with_prefix[task_index] = prefixed_task

    print(f"Prefix examples:")
    for i, task in list(tasks.items())[:3]:
        print(f"  [{i}] {prefix}{task}")

    # Initialize T5 encoder
    print(f"\nLoading T5 encoder from {wan_path} on {device}...")
    encoder = load_t5_encoder(wan_path, device)

    # Get episode files and generate embeddings
    data_dir = dataset_path / "data" / "chunk-000"
    episode_files = sorted(data_dir.glob("episode_*.parquet"))

    import pandas as pd

    for pq_file in tqdm(episode_files, desc="Generating T5 embeddings"):
        ep_name = pq_file.stem

        # Read task_index from parquet
        df = pd.read_parquet(pq_file)
        task_index = int(df['task_index'].iloc[0]) if 'task_index' in df.columns else 0

        # Get prefixed instruction
        if task_index in tasks_with_prefix:
            instruction = tasks_with_prefix[task_index]
        else:
            instruction = tasks.get(task_index, "")
            if instruction:
                instruction = f"{prefix}{instruction}"

        # Output path
        output_path = t5_dir / f"{ep_name}.pt"

        if not overwrite and output_path.exists():
            continue

        # Generate embedding
        emb = encode_t5(encoder, instruction, device)
        torch.save(emb, output_path)

    print(f"\nDone! T5 embeddings saved to {t5_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate T5 embeddings for LIBERO with prompt prefix")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to libero_*_no_noops_lerobot directory")
    parser.add_argument("--wan_path", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B directory containing T5 model")
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX,
                        help="Prompt prefix to prepend to each instruction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Skip existing embeddings")

    args = parser.parse_args()

    regenerate_t5_embeddings(
        dataset_dir=args.dataset_dir,
        wan_path=args.wan_path,
        prefix=args.prefix,
        device=args.device,
        overwrite=not args.no_overwrite,
    )
