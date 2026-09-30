#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Analysis Module for the EWAM Trimodal MoT Model.

Captures attention weights during inference by monkey-patching
flash_attention in wan/modules/attention_mask.py.

Two types of attention are captured:
1. Joint self-attention (attn_mask is not None): trimodal MoT [Video|Action|VLM]
2. T5 cross-attention (attn_mask is None, k_lens is not None): Video→Text

Usage:
    Set env var ATTENTION_ANALYSIS_DIR to enable, then run eval normally.
    Set ATTENTION_ANALYSIS_MAX_STEPS to limit captures (0 = unlimited).
"""

import os
import sys
import math
import json
import logging
import torch
import numpy as np
from pathlib import Path
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Module path for the flash_attention to patch
# EWAM repo: wan/ is at the project root (sibling of this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # EWAM repo root (this file lives in eval_scripts/)
_WAN_MODULES = str((_PROJECT_ROOT / "wan" / "modules").resolve())


def _get_attn_module():
    """Get the attention_mask module that contains flash_attention."""
    import wan.modules.attention_mask as attn_module
    return attn_module


def _get_all_flash_attention_refs():
    """
    Find all modules that have imported flash_attention.

    model_mask.py does `from .attention_mask import flash_attention`,
    which creates a separate reference. We need to patch all of them.
    """
    import wan.modules.attention_mask as attn_module
    refs = {'wan.modules.attention_mask': attn_module}

    # Patch model_mask.py too (it does `from .attention_mask import flash_attention`)
    try:
        import wan.modules.model_mask as model_mask_module
        if hasattr(model_mask_module, 'flash_attention'):
            refs['wan.modules.model_mask'] = model_mask_module
    except ImportError:
        pass

    # Also check any other module in sys.modules that might have flash_attention
    for name, mod in list(sys.modules.items()):
        if 'attention_mask' in name or 'model_mask' in name:
            if mod is not attn_module and hasattr(mod, 'flash_attention'):
                refs[name] = mod

    return refs


class AttentionCapture:
    """
    Context manager that patches flash_attention to capture attention weights.

    When active, replaces SDPA/flash_attn with manual softmax(QK^T/sqrt(d)) @ V
    computation, storing the weight matrices for analysis.

    For joint self-attention (attn_mask is not None):
        - Token layout: [Video (L_v) | Action (L_a) | VLM (L_vlm)]
        - Captures full [B, N, L_total, L_total] weight matrix

    For T5 cross-attention (attn_mask is None, k_lens is not None):
        - Video queries attend to T5 text keys
        - Captures [B, N, L_video, L_text] weight matrix
    """

    def __init__(self, save_dir, max_steps=0):
        """
        Args:
            save_dir: Directory to save captured weights and plots
            max_steps: Max number of inference_steps to capture (0 = unlimited)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps  # 0 = unlimited
        self.inference_step_idx = 0  # Which inference_step we're on
        self.captured = []  # List of dicts for current inference_step
        self.original_flash_attention = None
        self.enabled = False

        # Track whether this is joint (attn_mask) or cross (k_lens) attention
        self._call_count = 0  # Within one inference_step

    def __enter__(self):
        """Patch flash_attention in all modules that imported it."""
        refs = _get_all_flash_attention_refs()
        self.original_flash_attention = {}
        for name, mod in refs.items():
            self.original_flash_attention[name] = mod.flash_attention
            mod.flash_attention = self._patched_flash_attention
        self.enabled = True
        self.captured = []
        self._call_count = 0
        logger.info(f"AttentionCapture: patched flash_attention in {list(refs.keys())} for step {self.inference_step_idx}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original flash_attention and save results."""
        refs = _get_all_flash_attention_refs()
        for name, mod in refs.items():
            if name in self.original_flash_attention:
                mod.flash_attention = self.original_flash_attention[name]
        self.enabled = False

        # Compute stats from captured weights and save (small, KB-level)
        if self.captured:
            stats = self._compute_stats()
            save_path = self.save_dir / f"inference_step_{self.inference_step_idx:04d}.pt"
            torch.save({
                'step_idx': self.inference_step_idx,
                'stats': stats,
            }, str(save_path))
            logger.info(
                f"AttentionCapture: saved {len(stats)} attention stats "
                f"to {save_path}"
            )

        self.captured = []
        self.inference_step_idx += 1
        return False  # Don't suppress exceptions

    def _compute_stats(self):
        """
        Compute block-level statistics from captured attention weights.

        For joint self-attention (attn_mask is not None):
            Token layout: [Video (L_v) | Action (L_a) | VLM (L_vlm)]
            Detect boundaries from weight pattern, compute:
            - A→V, A→A, A→VLM mean attention (over heads and query tokens)
            - Per-head A→V, A→A, A→VLM distribution

        Returns a list of small dicts (no large tensors).
        """
        stats = []
        for cap in self.captured:
            w = cap['weights']  # [B, N, Lq, Lk]
            B, N, Lq, Lk = w.shape

            if cap['type'] == 'joint':
                # Detect L_v, L_a, L_vlm from weight pattern
                w_mean = w[0].mean(0)  # [Lq, Lk]
                nonzero_per_row = (w_mean > 1e-8).sum(dim=-1)  # [Lq]

                L_v = int(nonzero_per_row[0].item()) if Lq > 0 else 0
                L_vlm = int(nonzero_per_row[-1].item()) if Lq > 1 else 0
                L_a = Lk - L_v - L_vlm

                # Action rows: [L_v : L_v + L_a]
                if L_a > 0 and L_v >= 0 and L_vlm >= 0:
                    action_w = w[0, :, L_v:L_v + L_a, :]  # [N, L_a, Lk]

                    # Mean over query tokens, then mean over heads
                    a_to_v = action_w[:, :, :L_v].sum(dim=-1).mean().item() if L_v > 0 else 0.0
                    a_to_a = action_w[:, :, L_v:L_v + L_a].sum(dim=-1).mean().item() if L_a > 0 else 0.0
                    a_to_vlm = action_w[:, :, L_v + L_a:].sum(dim=-1).mean().item() if L_vlm > 0 else 0.0

                    total = a_to_v + a_to_a + a_to_vlm
                    if total > 0:
                        a_to_v_pct = a_to_v / total
                        a_to_a_pct = a_to_a / total
                        a_to_vlm_pct = a_to_vlm / total
                    else:
                        a_to_v_pct = a_to_a_pct = a_to_vlm_pct = 0.0

                    # Per-head distribution [N, 3]
                    head_to_v = action_w[:, :, :L_v].sum(dim=-1).mean(dim=1) if L_v > 0 else torch.zeros(N)
                    head_to_a = action_w[:, :, L_v:L_v + L_a].sum(dim=-1).mean(dim=1) if L_a > 0 else torch.zeros(N)
                    head_to_vlm = action_w[:, :, L_v + L_a:].sum(dim=-1).mean(dim=1) if L_vlm > 0 else torch.zeros(N)
                    head_totals = (head_to_v + head_to_a + head_to_vlm).clamp(min=1e-8)
                    per_head = torch.stack([
                        head_to_v / head_totals,
                        head_to_a / head_totals,
                        head_to_vlm / head_totals,
                    ], dim=1).cpu().numpy().tolist()  # [N, 3]
                else:
                    a_to_v_pct = a_to_a_pct = a_to_vlm_pct = 0.0
                    per_head = [[0.0, 0.0, 0.0]] * N

                stats.append({
                    'type': 'joint',
                    'call_idx': cap['call_idx'],
                    'L_v': L_v,
                    'L_a': L_a,
                    'L_vlm': L_vlm,
                    'A_to_V': a_to_v_pct,
                    'A_to_A': a_to_a_pct,
                    'A_to_VLM': a_to_vlm_pct,
                    'per_head': per_head,
                })
            else:
                # Cross attention: just store mean weight per head
                # w: [B, N, Lq, Lk] — too large to store, just summary stats
                w_flat = w[0].mean(dim=1)  # [Lq, Lk], average over heads
                # Top-10 attended key positions
                topk_vals, topk_idx = w_flat.mean(dim=0).topk(min(10, Lk))
                stats.append({
                    'type': 'cross',
                    'call_idx': cap['call_idx'],
                    'Lq': Lq,
                    'Lk': Lk,
                    'top_key_indices': topk_idx.cpu().numpy().tolist(),
                    'top_key_weights': topk_vals.cpu().numpy().tolist(),
                    'mean_weight': w_flat.mean().item(),
                })

        # Clear raw weights to free memory
        self.captured = []
        return stats

    def _patched_flash_attention(
        self, q, k, v, q_lens=None, k_lens=None, attn_mask=None,
        dropout_p=0., softmax_scale=None, q_scale=None, causal=False,
        window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
        version=None,
    ):
        """
        Replacement for flash_attention that captures attention weights.

        For joint self-attention (attn_mask is not None):
            Uses manual softmax computation instead of SDPA.

        For T5 cross-attention (attn_mask is None, k_lens is not None):
            Uses manual computation instead of flash_attn library.
        """
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        out_dtype = q.dtype

        # ── Joint self-attention path (attn_mask is not None) ──────────
        if attn_mask is not None:
            return self._manual_attention_with_mask(
                q, k, v, attn_mask, dropout_p, dtype, out_dtype
            )

        # ── T5 cross-attention path (attn_mask is None) ────────────────
        # This path normally goes through flash_attn_varlen_func
        # We replace it with manual computation to capture weights
        if k_lens is not None:
            return self._manual_attention_cross(
                q, k, v, q_lens, k_lens, q_scale, dtype, out_dtype
            )

        # Fallback: shouldn't happen in normal inference, use original
        original = list(self.original_flash_attention.values())[0]
        return original(
            q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens, attn_mask=attn_mask,
            dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
            causal=causal, window_size=window_size, deterministic=deterministic,
            dtype=dtype, version=version,
        )

    def _manual_attention_with_mask(self, q, k, v, attn_mask, dropout_p, dtype, out_dtype):
        """Manual attention for joint self-attention (with attn_mask)."""
        b, lq, lk = q.size(0), q.size(1), k.size(1)

        # Trim mask to actual sizes
        attn_mask = attn_mask[:, :lq, :lk]
        attn_mask_sdpa = attn_mask.unsqueeze(1)  # [B, 1, Lq, Lk]

        # Transpose to [B, N, L, C]
        q_t = q.to(dtype).transpose(1, 2)
        k_t = k.to(dtype).transpose(1, 2)
        v_t = v.to(dtype).transpose(1, 2)

        # Manual attention: Q @ K^T / sqrt(d)
        scale = 1.0 / math.sqrt(q_t.size(-1))
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, N, Lq, Lk]

        # Apply mask: True=allow, False=mask -> fill masked with -inf
        scores = scores.masked_fill(~attn_mask_sdpa.bool(), float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N, Lq, Lk]

        # Apply attention to values
        out = torch.matmul(attn_weights, v_t)  # [B, N, Lq, C]
        out = out.transpose(1, 2).contiguous().type(out_dtype)  # [B, Lq, N, C]

        # Capture weights
        self.captured.append({
            'type': 'joint',
            'call_idx': self._call_count,
            'weights': attn_weights.detach().cpu().float(),  # [B, N, Lq, Lk]
            'Lq': lq,
            'Lk': lk,
            'num_heads': q.size(2),
            'head_dim': q.size(3),
        })
        self._call_count += 1

        return out

    def _manual_attention_cross(self, q, k, v, q_lens, k_lens, q_scale, dtype, out_dtype):
        """Manual attention for T5 cross-attention (no mask, with k_lens)."""
        b, lq, lk = q.size(0), q.size(1), k.size(1)

        # For cross-attention, q_lens is typically [lq]*b and k_lens is [lk]*b
        # We handle the simple case (all same length) for weight capture
        # Transpose to [B, N, L, C]
        q_t = q.to(dtype).transpose(1, 2)
        k_t = k.to(dtype).transpose(1, 2)
        v_t = v.to(dtype).transpose(1, 2)

        if q_scale is not None:
            q_t = q_t * q_scale

        # Manual attention
        scale = 1.0 / math.sqrt(q_t.size(-1))
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, N, Lq, Lk]

        # No mask for cross-attention
        attn_weights = torch.softmax(scores, dim=-1)  # [B, N, Lq, Lk]
        out = torch.matmul(attn_weights, v_t)  # [B, N, Lq, C]
        out = out.transpose(1, 2).contiguous().type(out_dtype)  # [B, Lq, N, C]

        # Capture weights
        self.captured.append({
            'type': 'cross',
            'call_idx': self._call_count,
            'weights': attn_weights.detach().cpu().float(),  # [B, N, Lq, Lk]
            'Lq': lq,
            'Lk': lk,
            'num_heads': q.size(2),
            'head_dim': q.size(3),
        })
        self._call_count += 1

        return out

    def generate_report(self):
        """
        Generate visualization report from all captured inference_steps.

        Call this after the episode is complete.
        Reads all inference_step_*.pt files (which contain pre-computed stats).
        """
        # Find all saved files
        files = sorted(self.save_dir.glob("inference_step_*.pt"))
        if not files:
            logger.warning("No attention capture files found for report")
            return

        logger.info(f"Generating attention analysis report from {len(files)} inference steps...")

        all_summaries = []

        for f in files:
            data = torch.load(str(f), weights_only=False)
            stats = data['stats']
            step_idx = data['step_idx']

            joint_stats = [s for s in stats if s['type'] == 'joint']
            cross_stats = [s for s in stats if s['type'] == 'cross']

            step_summary = self._build_step_summary(step_idx, joint_stats, cross_stats)
            all_summaries.append(step_summary)

        # Save summary JSON
        summary_path = self.save_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        # Generate plots
        self._plot_action_attention(all_summaries)
        self._plot_per_head(all_summaries, files)
        self._plot_trend(all_summaries)

        logger.info(f"Attention analysis report saved to {self.save_dir}")

    def _build_step_summary(self, step_idx, joint_stats, cross_stats):
        """
        Build summary from pre-computed joint attention stats.

        Joint attention call order within one inference_step:
            Per denoising step (2 steps), per layer (30 layers):
                joint self-attention (60 total)
        """
        num_layers = 30
        num_denoise = len(joint_stats) // num_layers if joint_stats else 0

        step_summary = {
            'step_idx': step_idx,
            'num_joint': len(joint_stats),
            'num_cross': len(cross_stats),
            'num_denoise': num_denoise,
            'layers': [],
        }

        for d in range(num_denoise):
            for layer in range(num_layers):
                idx = d * num_layers + layer
                if idx >= len(joint_stats):
                    break

                s = joint_stats[idx]
                step_summary['layers'].append({
                    'denoise_step': d,
                    'layer': layer,
                    'L_v': s['L_v'],
                    'L_a': s['L_a'],
                    'L_vlm': s['L_vlm'],
                    'A_to_V': s['A_to_V'],
                    'A_to_A': s['A_to_A'],
                    'A_to_VLM': s['A_to_VLM'],
                    'per_head': s['per_head'],
                })

        return step_summary

    def _plot_action_attention(self, all_summaries):
        """Plot Action→{V, A, VLM} attention distribution across layers."""
        if not all_summaries:
            return

        # Use first step's first denoise pass for the main heatmap
        step = all_summaries[0]
        num_layers = 30
        num_denoise = step['num_denoise']

        fig, axes = plt.subplots(num_denoise, 1, figsize=(12, 4 * num_denoise), squeeze=False)

        for d in range(num_denoise):
            ax = axes[d, 0]
            layers_data = [l for l in step['layers'] if l['denoise_step'] == d]
            if not layers_data:
                continue

            data = np.array([[l['A_to_V'], l['A_to_A'], l['A_to_VLM']] for l in layers_data])

            im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_ylabel('Target Expert')
            ax.set_xlabel('Layer')
            ax.set_title(f'Denoise Step {d}: Action Attention Distribution')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['→Video', '→Action', '→VLM'])
            ax.set_xticks(range(num_layers))
            ax.set_xticklabels(range(num_layers), fontsize=6)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        path = self.save_dir / "action_attention_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved action attention heatmap to {path}")

    def _plot_per_head(self, all_summaries, files):
        """Plot per-head attention distribution across all layers (denoise step 0, first inference step)."""
        if not all_summaries:
            return

        step = all_summaries[0]
        num_layers = 30
        d = 0  # denoise step 0

        # Collect per-head data [N, 3] for each layer
        all_per_head = []
        for layer in range(num_layers):
            target = [l for l in step['layers'] if l['denoise_step'] == d and l['layer'] == layer]
            if target:
                all_per_head.append(np.array(target[0]['per_head']))
            else:
                all_per_head.append(None)

        valid = [ph for ph in all_per_head if ph is not None]
        if not valid:
            return
        N = valid[0].shape[0]

        # Build [num_layers, N] matrices for each target
        data_v = np.zeros((num_layers, N))
        data_a = np.zeros((num_layers, N))
        data_vlm = np.zeros((num_layers, N))
        for layer, ph in enumerate(all_per_head):
            if ph is not None:
                data_v[layer] = ph[:, 0]
                data_a[layer] = ph[:, 1]
                data_vlm[layer] = ph[:, 2]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        titles = ['→Video', '→Action', '→VLM']
        matrices = [data_v, data_a, data_vlm]
        cmaps = ['Blues', 'Oranges', 'Greens']

        for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
            im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)
            ax.set_ylabel('Layer')
            ax.set_title(f'Per-Head Attention: {title} (Denoise Step {d})')
            if ax is axes[-1]:
                ax.set_xlabel('Head Index')
                ax.set_xticks(range(N))
                ax.set_xticklabels(range(N), fontsize=5)
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels(range(num_layers), fontsize=6)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        path = self.save_dir / "per_head_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-head heatmap to {path}")

    def _plot_trend(self, all_summaries):
        """Plot attention trend across inference steps for all layers, one row per denoise step."""
        if len(all_summaries) <= 1:
            return

        num_steps = len(all_summaries)
        num_layers = 30
        num_denoise = all_summaries[0].get('num_denoise', 1)

        fig, axes = plt.subplots(num_denoise, 3, figsize=(18, 4 * num_denoise), squeeze=False)
        targets = ['→Video', '→Action', '→VLM']
        cmaps = ['Blues', 'Oranges', 'Greens']

        for d in range(num_denoise):
            # Build [num_layers, num_steps] matrix for each target
            matrices = [np.zeros((num_layers, num_steps)) for _ in range(3)]
            for step_i, s in enumerate(all_summaries):
                for layer in range(num_layers):
                    target = [l for l in s['layers'] if l['denoise_step'] == d and l['layer'] == layer]
                    if target:
                        matrices[0][layer, step_i] = target[0]['A_to_V']
                        matrices[1][layer, step_i] = target[0]['A_to_A']
                        matrices[2][layer, step_i] = target[0]['A_to_VLM']

            for j, (mat, title, cmap) in enumerate(zip(matrices, targets, cmaps)):
                ax = axes[d, j]
                im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=1)
                ax.set_xlabel('Inference Step')
                ax.set_ylabel('Layer')
                ax.set_title(f'{title} (Denoise Step {d})')
                ax.set_yticks(range(num_layers))
                ax.set_yticklabels(range(num_layers), fontsize=6)
                plt.colorbar(im, ax=ax)

        plt.tight_layout()
        path = self.save_dir / "attention_trend.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved attention trend to {path}")
