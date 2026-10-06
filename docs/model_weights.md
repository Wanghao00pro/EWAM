# Model Weights and Configuration

**Released EWAM checkpoints** (stage-1 pretrain + the three stage-2 finetunes) are hosted on HuggingFace: [HaoWang00/EWAM](https://huggingface.co/HaoWang00/EWAM); see the [main README](../README.md) for the checkpoint list and download commands.

| Checkpoint | Usage |
|---|---|
| Pretrain (multi-source, stage-1) | Init for stage-2 finetune — `finetune.checkpoint_path` |
| RoboTwin c2r | RoboTwin 2.0 evaluation — `checkpoint_path` in `paths_config.yml` (see the [RoboTwin guide](robotwin/README.md#evaluation)) |
| RoboTwin in-domain | RoboTwin 2.0 evaluation — `checkpoint_path` in `paths_config.yml` (see the [RoboTwin guide](robotwin/README.md#evaluation)) |
| LIBERO | LIBERO evaluation — `CKPT` override of the eval launcher (see the [LIBERO guide](libero/README.md#evaluation)) |

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
