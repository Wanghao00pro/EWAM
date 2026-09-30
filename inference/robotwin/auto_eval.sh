#!/bin/bash
# Auto evaluation script for the Ewam_dim16 policy on the RoboTwin platform.
# Runs all tasks in tasks_all.txt in parallel across available GPUs.
#
# Deploy layout (see README "RoboTwin → Evaluation"):
#   RoboTwin/policy/Ewam_dim16/
#     ├── auto_eval.sh            (this file)
#     ├── eval.sh                 (single-task debug)
#     ├── paths_config.yml        (copy from paths_config.example.yml, then edit)
#     ├── deploy_policy.py        (EWAM policy — from EWAM/inference/robotwin/)
#     ├── deploy_policy.yml
#     ├── requirements.txt
#     ├── tasks_all.txt           (50 RoboTwin2.0 tasks)
#     └── paths_config.example.yml

echo "Starting Ewam evaluation on RoboTwin at $(date)"

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
LOG_DIR_CFG=$(grep "^log_dir:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Optional configurations
TASK_CONFIG=$(grep "^task_config:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
TASKS_FILE=$(grep "^tasks_file:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

# Default values if not in config
POLICY_NAME=${POLICY_NAME:-"Ewam_dim16"}
TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
TASKS_FILE=${TASKS_FILE:-"tasks_all.txt"}

# Parse GPU IDs from config (if specified)
GPU_IDS_STR=$(grep "^gpu_ids:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *\[\(.*\)\]/\1/' | tr -d ' ')
if [ -n "$GPU_IDS_STR" ] && [ "$GPU_IDS_STR" != "[]" ] && [ "$GPU_IDS_STR" != "" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_IDS_STR"
else
    GPU_IDS=()  # Empty = auto-detect
fi

# ============================================================================
# Validation
# ============================================================================
if [ -z "$ROBOTWIN_ROOT" ]; then
    echo "Error: robotwin_root is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: conda_env is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$WAN_PATH" ]; then
    echo "Error: wan_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$VLM_PATH" ]; then
    echo "Error: vlm_path is not set in $CONFIG_FILE"
    exit 1
fi

if [ -z "$EWAM_ROOT" ]; then
    echo "Error: ewam_root is not set in $CONFIG_FILE"
    echo "(EWAM repository checkout that contains models/ewam.py —"
    echo " the deployed policy imports the EWAM package from there)"
    exit 1
fi

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

# Optional training config for the checkpoint (architecture only). If empty,
# deploy_policy.py falls back to its default: $EWAM_ROOT/configs/ewam_robotwin.yaml
EXTRA_ARGS=()
if [ -n "$CONFIG_PATH" ]; then
    EXTRA_ARGS+=(--config_path "$CONFIG_PATH")
fi

# Export for deploy_policy.py's EWAM package import
export EWAM_ROOT

cd "$ROBOTWIN_ROOT" || exit 1

# Activate conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found."
    exit 1
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

# Create logs directory
LOG_DIR="${LOG_DIR_CFG:-${ROBOTWIN_ROOT}/eval_logs/ewam_auto_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

# Load tasks
TASKS_PATH="${POLICY_DIR}/${TASKS_FILE}"
if [ ! -f "$TASKS_PATH" ]; then
    echo "Error: Tasks file not found: $TASKS_PATH"
    exit 1
fi
mapfile -t tasks < <(grep -v '^\s*$' "$TASKS_PATH")

if [ ${#tasks[@]} -eq 0 ]; then
    echo "Error: No tasks found."
    exit 1
fi

# Auto-detect GPUs if not specified
if [ ${#GPU_IDS[@]} -eq 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
        echo "Auto-detected ${#GPU_IDS[@]} GPUs: ${GPU_IDS[*]}"
    else
        echo "Warning: nvidia-smi not found, using GPU 0"
        GPU_IDS=(0)
    fi
fi

# Memory thresholds (in MiB) for task allocation
GPU_MEMORY_HIGH=90000      # < 90GB: 1 task
GPU_MEMORY_MID=45000       # < 50GB: 2 tasks
GPU_MEMORY_LOW=3000        # < 5GB: 3 tasks

echo -e "\n\033[33m=== Evaluation Configuration ===\033[0m"
echo "RoboTwin Root: $ROBOTWIN_ROOT"
echo "EWAM Root:     $EWAM_ROOT"
echo "Policy Dir:    $POLICY_DIR"
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "WAN Path:      $WAN_PATH"
echo "VLM Path:      $VLM_PATH"
echo "Policy:        $POLICY_NAME"
echo "Task Config:   $TASK_CONFIG"
echo "Tasks:         ${#tasks[@]}"
echo "GPUs:          ${GPU_IDS[*]}"
echo "GPU Memory - High: ${GPU_MEMORY_HIGH}MiB (1 task), Mid: ${GPU_MEMORY_MID}MiB (2 tasks), Low: ${GPU_MEMORY_LOW}MiB (3 tasks)"
echo "Seed:          $SEED"
echo "Log Dir:       $LOG_DIR"
echo "================================"

# GPU management - Track multiple PIDs per GPU using arrays
declare -A gpu_pids

for gpu_id in "${GPU_IDS[@]}"; do
    gpu_pids[$gpu_id]=""
done

# Get GPU memory usage in MiB for a specific GPU
get_gpu_memory_used() {
    local gpu_id=$1
    nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
}

is_running() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

get_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            local mem_used=$(get_gpu_memory_used $gpu_id)

            # Count running tasks on this GPU
            local task_count=0
            local pids_str="${gpu_pids[$gpu_id]}"
            if [ -n "$pids_str" ]; then
                for pid in $pids_str; do
                    if is_running "$pid"; then
                        ((task_count++))
                    fi
                done
            fi

            # Check available slots based on memory
            if [ -n "$mem_used" ]; then
                # Slot 1: < 90GB (always available at start)
                if [ "$mem_used" -lt "$GPU_MEMORY_HIGH" ]; then
                    if [ $task_count -eq 0 ]; then
                        echo "$gpu_id:1"
                        return 0
                    # Slot 2: < 50GB and already have 1 task
                    elif [ "$mem_used" -lt "$GPU_MEMORY_MID" ] && [ $task_count -eq 1 ]; then
                        echo "$gpu_id:2"
                        return 0
                    # Slot 3: < 5GB and already have 2 tasks
                    elif [ "$mem_used" -lt "$GPU_MEMORY_LOW" ] && [ $task_count -eq 2 ]; then
                        echo "$gpu_id:3"
                        return 0
                    fi
                fi
            fi
        done
        sleep 2
    done
}

show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local bar_length=50
    local filled=$((percent * bar_length / 100))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%$((bar_length - filled))s" | tr ' ' ' '
    printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

# Launch tasks
pids=()
completed=0
total=${#tasks[@]}

echo -e "\n\033[32mLaunching evaluation tasks...\033[0m"

for task in "${tasks[@]}"; do
    # Parse GPU selection (format: gpu_id:slot_number)
    gpu_selection=$(get_free_gpu)
    gpu_id=$(echo $gpu_selection | cut -d: -f1)
    slot=$(echo $gpu_selection | cut -d: -f2)
    mem_used=$(get_gpu_memory_used $gpu_id)
    ckpt_setting="${CHECKPOINT_PATH}"
    log_file="${LOG_DIR}/${task}.log"

    echo -e "\033[36m-> Task: $task | GPU: $gpu_id | Slot: $slot | Memory: ${mem_used}MiB\033[0m"

    (
        export CUDA_VISIBLE_DEVICES=$gpu_id

        PYTHONWARNINGS=ignore::UserWarning \
        python script/eval_policy.py \
            --config "policy/${POLICY_NAME}/deploy_policy.yml" \
            --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "${ckpt_setting}" \
            --seed "${SEED}" \
            --policy_name "${POLICY_NAME}" \
            --log_dir "${LOG_DIR}" \
            --wan_path "${WAN_PATH}" \
            --vlm_path "${VLM_PATH}" \
            "${EXTRA_ARGS[@]}" \
            > "$log_file" 2>&1

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "Task $task completed successfully" >> "$log_file"
        else
            echo "Task $task failed with exit code $exit_code" >> "$log_file"
        fi
    ) &

    pid=$!
    # Append PID to the GPU's PID list (space-separated)
    if [ -z "${gpu_pids[$gpu_id]}" ]; then
        gpu_pids[$gpu_id]="$pid"
    else
        gpu_pids[$gpu_id]="${gpu_pids[$gpu_id]} $pid"
    fi
    pids+=($pid)
    sleep 1
done

echo -e "\n\033[33mWaiting for completion...\033[0m"

for pid in "${pids[@]}"; do
    wait "$pid"
    ((completed++))
    show_progress $completed $total
done

echo -e "\n\033[32mAll tasks completed!\033[0m"

# Generate summary
summary="${LOG_DIR}/evaluation_summary.txt"

cat > "$summary" << EOF
Ewam Evaluation Summary
=======================================
Date: $(date)
Host: $(hostname)
RoboTwin: $ROBOTWIN_ROOT
EWAM: $EWAM_ROOT
Checkpoint: $CHECKPOINT_PATH
WAN Path: $WAN_PATH
VLM Path: $VLM_PATH
Policy: $POLICY_NAME
Task Config: $TASK_CONFIG
Seed: $SEED
Total Tasks: $total
GPUs: ${GPU_IDS[*]}

Task Results:
-------------
EOF

success=0
failed=0

for task in "${tasks[@]}"; do
    log_file="${LOG_DIR}/${task}.log"

    if [ ! -f "$log_file" ]; then
        echo "  ? $task: LOG NOT FOUND" >> "$summary"
        ((failed++))
    elif grep -q "completed successfully\|Episode.*completed" "$log_file" 2>/dev/null; then
        echo "  OK $task: SUCCESS" >> "$summary"
        ((success++))
    else
        echo "  XX $task: FAILED" >> "$summary"
        ((failed++))
    fi
done

cat >> "$summary" << EOF

Summary Statistics:
-------------------
OK (Success): $success
XX (Failed): $failed
Total: $total
Success Rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%

Logs: $LOG_DIR
EOF

echo -e "\n=== Summary ==="
echo "OK (Success): $success"
echo "XX (Failed): $failed"
echo "Success Rate: $(awk "BEGIN {printf \"%.1f\", $success * 100.0 / $total}")%"
echo "Summary: $summary"

if [ $failed -eq 0 ]; then
    echo "All tasks passed!"
    exit 0
else
    echo "Check logs for failures."
    exit 1
fi
