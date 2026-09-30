#!/bin/bash
# Single-task evaluation script for the Ewam_dim16 policy on the RoboTwin
# platform (debugging helper for auto_eval.sh — same paths_config.yml, one task, foreground).

# ============================================================================
# Single Task Configuration - MODIFY THESE
# ============================================================================
TASK_NAME="place_can_basket"    # Change this to the task you want to test
GPU_ID='0'                      # GPU to use

# ============================================================================
# Script starts here
# ============================================================================
echo "Starting Ewam evaluation at $(date)"

# Get script directory (policy/Ewam_dim16/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"

# ============================================================================
# Load Configuration from paths_config.yml
# ============================================================================
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please copy paths_config.example.yml to paths_config.yml and edit the paths."
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"

# Parse YAML (remove comments and extra whitespace)
ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
EWAM_ROOT=$(grep "^ewam_root:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONFIG_PATH=$(grep "^config_path:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
POLICY_NAME=$(grep "^policy_name:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values
POLICY_NAME=${POLICY_NAME:-"Ewam_dim16"}
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}

# ============================================================================
# Validation
# ============================================================================
for var_name in ROBOTWIN_ROOT CONDA_ENV CHECKPOINT_PATH WAN_PATH VLM_PATH EWAM_ROOT; do
    if [ -z "${!var_name}" ]; then
        echo "Error: ${var_name} is not set in $CONFIG_FILE"
        exit 1
    fi
done

if [ ! -d "$ROBOTWIN_ROOT" ]; then
    echo "Error: RoboTwin root not found: $ROBOTWIN_ROOT"
    exit 1
fi

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$WAN_PATH" ]; then
    echo "Error: WAN path not found: $WAN_PATH"
    exit 1
fi

if [ ! -d "$VLM_PATH" ]; then
    echo "Error: VLM path not found: $VLM_PATH"
    exit 1
fi

if [ ! -f "$EWAM_ROOT/models/ewam.py" ]; then
    echo "Error: EWAM repository not found at: $EWAM_ROOT (missing models/ewam.py)"
    exit 1
fi

# Optional training config for the checkpoint (architecture only)
EXTRA_ARGS=()
if [ -n "$CONFIG_PATH" ]; then
    EXTRA_ARGS+=(--config_path "$CONFIG_PATH")
fi

export EWAM_ROOT

cd "$ROBOTWIN_ROOT" || exit 1

# Activate conda
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

# Set environment
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
LOG_DIR="${ROBOTWIN_ROOT}/eval_logs/ewam_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ckpt_setting="${CHECKPOINT_PATH}"
log_file="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Ewam Evaluation Configuration"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "----------------------------------------------------------------"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "EWAM Root:         $EWAM_ROOT"
echo "Policy Dir:        $POLICY_DIR"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "WAN Path:          $WAN_PATH"
echo "VLM Path:          $VLM_PATH"
echo "Task Config:       $TASK_CONFIG"
echo "Seed:              $SEED"
echo "Log File:          $log_file"
echo "================================================================"
echo ""

# Run evaluation
echo "Starting evaluation..."

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/${POLICY_NAME}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${SEED}" \
    --policy_name "${POLICY_NAME}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log_file"

exit_code=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $exit_code -eq 0 ]; then
    echo "Task $TASK_NAME completed successfully"
    echo "================================================================"
    exit 0
else
    echo "Task $TASK_NAME failed with exit code $exit_code"
    echo "================================================================"
    echo "Log file: $log_file"
    exit 1
fi
