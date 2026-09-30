#!/bin/bash
# NOTE: no `set -e` here. The scheduler is a long-running loop where individual
# commands (grep with no match, ((var++)) when var==0, [ -s file ] && ...) legally
# return non-zero; `set -e` would abort the whole run on the first such case.
# `set -u` catches undefined variables; pipefail keeps pipeline failures visible.
set -uo pipefail

# EWAM dim16 (Horizontal) — LIBERO 4-suite parallel evaluation.
#
# Evaluates all 4 LIBERO suites (libero_spatial, libero_object, libero_goal, libero_10;
# 10 tasks each = 40 tasks) in parallel across GPUs using a dynamic tmux-pane scheduler
# Each GPU runs up to
# MAX_TASKS_PER_GPU concurrent tasks; a least-loaded-GPU scheduler dispatches new tasks
# as capacity frees up. Task completion is detected by the result JSON file appearing;
# failures (non-zero exit, no result JSON) abort the run.
#
# Horizontal variant: image_concat_mode="horizontal" (agentview | wrist side-by-side),
# 224x448.
#
# dim16 conversions (state 8->16 pad, action 16->7 restore) and [-1,1] min/max normalization
# (aggregated across all 4 suites) are handled inside the eval script.
#
# 4-suite batch evaluation
#
# Usage:
#   bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
#   # Override defaults via env vars:
#   CKPT=... STATS=... CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_TASKS_PER_GPU=3 NUM_TRIALS=50 \
#     SUITES="libero_object libero_10" bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
#   # Or evaluate a single suite / fewer trials:
#   NUM_TRIALS=10 SUITES="libero_object" bash scripts/run_ewam_libero_eval_horizontal_dim16.sh
#

# Resolve the EWAM project root (parent of this scripts/ directory)
EWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Third-party LIBERO benchmark (see README for installation)
LIBERO_ROOT="${LIBERO_ROOT:-}"
if [ -z "$LIBERO_ROOT" ]; then
    echo "ERROR: LIBERO_ROOT is not set. Export it to your LIBERO checkout first, e.g.:"
    echo "  export LIBERO_ROOT=/path/to/LIBERO-master"
    exit 1
fi
# Conda environment to run evals in. Set CONDA_ENV=<name> to activate one; leave it
# empty to run with whatever python is currently active. Empty by default.
CONDA_ENV="${CONDA_ENV:-}"
CONDA_BASE="$(conda info --base 2>/dev/null)"

cd "$EWAM_ROOT"
if [ -n "$CONDA_ENV" ] && [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export PYTHONPATH="${EWAM_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Default GPUs to use. Override from command line: CUDA_VISIBLE_DEVICES=0,1 bash ...
# (unset/empty falls back to NUM_GPUS starting from 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

# Paper-result checkpoint: trained with the plain loss (train_ewam.py +
# configs/ewam_libero.yaml). Override with CKPT=... / STATS=...
DEFAULT_CKPT="/path/to/ewam_weights/libero/pytorch_model/mp_rank_00_model_states.pt"

DEFAULT_STATS="/path/to/ewam_weights/libero/dataset_stats.json"

CKPT="${CKPT:-$DEFAULT_CKPT}"
STATS="${STATS:-$DEFAULT_STATS}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TASKS_PER_SUITE="${NUM_TASKS_PER_SUITE:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$EWAM_ROOT/evaluate_results/ewam_libero_horizontal_dim16/${RUN_ID}}"

# GPU selection: CUDA_VISIBLE_DEVICES takes priority over NUM_GPUS.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ... use those exact physical GPU ids
#   CUDA_VISIBLE_DEVICES unset   ... use NUM_GPUS starting from 0 (default: 8 -> 0..7)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    AVAILABLE_GPUS="$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS="${NUM_GPUS:-8}"
    AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
fi
export NUM_GPUS
# Parse AVAILABLE_GPUS into an array (real physical GPU ids)
IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

EVAL_SCRIPT="eval_scripts/eval_ewam_libero_single_horizontal_dim16.py"

# Validate checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "Available checkpoints:"
    find "$(dirname "$(dirname "$(dirname "$CKPT")")")" -maxdepth 1 -name "checkpoint_step_*" -o -name "best_action_l2" 2>/dev/null || true
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build task list: one "suite,task_id" line per task
TASK_FILE="$OUTPUT_DIR/tasks.txt"
: > "$TASK_FILE"
for suite in $SUITES; do
    for ((tid=0; tid<NUM_TASKS_PER_SUITE; tid++)); do
        echo "${suite},${tid}" >> "$TASK_FILE"
    done
done
TOTAL_TASKS=$(wc -l < "$TASK_FILE")

# Max concurrent tasks per GPU (tmux pane pool). Override: MAX_TASKS_PER_GPU=2 bash ...
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-3}"

echo "=========================================="
echo "EWAM dim16 (Horizontal) LIBERO 4-suite eval"
echo "=========================================="
echo "Checkpoint:   $CKPT"
echo "Dataset stats: $STATS (falls back to 4-suite aggregation if absent)"
echo "Suites:       $SUITES"
echo "Tasks/suite:  $NUM_TASKS_PER_SUITE  (total: $TOTAL_TASKS)"
echo "Trials/task:  $NUM_TRIALS"
echo "GPUs:         ${GPU_ARRAY[*]}  (${NUM_GPUS} total)"
echo "Max tasks/GPU: $MAX_TASKS_PER_GPU  (concurrent tasks per GPU, tmux pane pool)"
echo "Output dir:   $OUTPUT_DIR"
echo "=========================================="

cp "$TASK_FILE" "$OUTPUT_DIR/tasks.txt.bak"

# ============================================================================
# Dynamic GPU scheduler.
# Each GPU runs up to MAX_TASKS_PER_GPU concurrent tasks in tmux panes. A task is
# considered complete when its result JSON appears; failed tasks (non-zero exit,
# no result JSON) are recorded and abort the run.
# ============================================================================

SESSION_NAME="ewam_eval_${RUN_ID}"
GPU_LOAD_FILE="$OUTPUT_DIR/gpu_load.txt"
TASK_GPU_MAP_FILE="$OUTPUT_DIR/task_gpu_map.txt"
TASK_STATUS_DIR="$OUTPUT_DIR/task_status"
TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"

mkdir -p "$TASK_STATUS_DIR" "$TASK_LOG_DIR"
: > "$FAILED_TASKS_FILE"

# Initialize GPU load tracking: gpu_id -> current task count
init_gpu_load_tracking() {
    > "$GPU_LOAD_FILE"
    > "$TASK_GPU_MAP_FILE"
    for gpu in "${GPU_ARRAY[@]}"; do
        echo "$gpu:0" >> "$GPU_LOAD_FILE"
    done
    echo "GPU load tracking initialized: $GPU_LOAD_FILE"
}

get_gpu_load() {
    local gpu_id=$1
    local load
    load=$(grep "^$gpu_id:" "$GPU_LOAD_FILE" | cut -d: -f2)
    echo "${load:-0}"
}

update_gpu_load() {
    local gpu_id=$1 new_load=$2
    local temp_file="$GPU_LOAD_FILE.tmp"
    if [ -f "$GPU_LOAD_FILE" ]; then
        grep -v "^${gpu_id}:" "$GPU_LOAD_FILE" > "$temp_file" 2>/dev/null || true
    else
        > "$temp_file"
    fi
    echo "${gpu_id}:${new_load}" >> "$temp_file"
    mv "$temp_file" "$GPU_LOAD_FILE"
}

increment_gpu_load() {
    local gpu_id=$1
    local new_load=$(( $(get_gpu_load $gpu_id) + 1 ))
    update_gpu_load $gpu_id $new_load
    echo $new_load
}

decrement_gpu_load() {
    local gpu_id=$1
    local new_load=$(( $(get_gpu_load $gpu_id) - 1 ))
    [ $new_load -lt 0 ] && new_load=0
    update_gpu_load $gpu_id $new_load
    echo $new_load
}

# Find the least-loaded GPU that still has capacity (< MAX_TASKS_PER_GPU)
find_least_loaded_gpu() {
    local min_load=999999 best_gpu=""
    for gpu in "${GPU_ARRAY[@]}"; do
        local load
        load=$(get_gpu_load $gpu)
        if [ $load -lt $min_load ] && [ $load -lt $MAX_TASKS_PER_GPU ]; then
            min_load=$load
            best_gpu=$gpu
        fi
    done
    echo "$best_gpu"
}

record_task_gpu_mapping() {
    local suite=$1 task_id=$2 gpu_id=$3
    echo "$suite,$task_id:$gpu_id" >> "$TASK_GPU_MAP_FILE"
}

get_task_gpu() {
    local suite=$1 task_id=$2
    local mapping
    mapping=$(grep "^$suite,$task_id:" "$TASK_GPU_MAP_FILE" | cut -d: -f2)
    echo "${mapping:-}"
}

remove_task_gpu_mapping() {
    local suite=$1 task_id=$2
    local temp_file="$TASK_GPU_MAP_FILE.tmp"
    grep -v "^$suite,$task_id:" "$TASK_GPU_MAP_FILE" > "$temp_file" 2>/dev/null || true
    mv "$temp_file" "$TASK_GPU_MAP_FILE"
}

mark_task_failed() {
    local suite=$1 task_id=$2 gpu_id=$3 rc=$4 log_file=$5
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$ts,$suite,$task_id,gpu=$gpu_id,rc=$rc,log=$log_file" >> "$FAILED_TASKS_FILE"
}

# EWAM eval writes result JSON to: $OUTPUT_DIR/$suite/task${task_id}_results.json
# (no gpu prefix in filename; task_id is unique per suite so no collision).
result_file_for() {
    local suite=$1 task_id=$2
    echo "$OUTPUT_DIR/$suite/task${task_id}_results.json"
}

# Launch a single task in a tmux pane. When the task exits, write a status file
# so the scheduler can detect completion/failure promptly.
# Conda activation prefix for tmux panes (empty when CONDA_ENV is unset).
ACTIVATE_CMD=""
if [ -n "$CONDA_ENV" ] && [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    ACTIVATE_CMD="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$CONDA_ENV' && "
fi
launch_task_on_pane() {
    local suite=$1 task_id=$2 gpu_id=$3 pane_info=$4
    local status_file="$TASK_STATUS_DIR/${suite}_task${task_id}.status"
    local log_file="$TASK_LOG_DIR/${suite}_task${task_id}_gpu${gpu_id}.log"
    rm -f "$status_file"
    echo "[$(date '+%H:%M:%S')] Launching: $suite task_id=$task_id on GPU$gpu_id (pane $pane_info)"
    # pane_info is a tmux %id (e.g. %1276). The pane's shell is idle (cleanup only returns
    # a pane to the free stack after its previous task wrote the status file, i.e. the
    # `python ...; rc=$?; echo STATUS` chain has finished). So we can send-keys directly.
    # Clear any leftover prompt, then send the task command.
    tmux send-keys -t "$pane_info" "clear" C-m 2>/dev/null
    sleep 0.3
    tmux send-keys -t "$pane_info" "${ACTIVATE_CMD}\
        export PYTHONPATH='${EWAM_ROOT}:${LIBERO_ROOT}:\$PYTHONPATH' && \
        export MUJOCO_GL=osmesa && \
        export HYDRA_FULL_ERROR=1 && export TOKENIZERS_PARALLELISM=false && \
        cd '${EWAM_ROOT}' && \
        CUDA_VISIBLE_DEVICES=$gpu_id python $EVAL_SCRIPT \
        ckpt='$CKPT' EVALUATION.dataset_stats_path='$STATS' \
        EVALUATION.task_suite_name=$suite EVALUATION.task_id=$task_id \
        EVALUATION.num_trials=$NUM_TRIALS EVALUATION.output_dir='$OUTPUT_DIR' \
        > '$log_file' 2>&1; \
        rc=\$?; \
        if [ \$rc -eq 0 ] && [ -f '$(result_file_for $suite $task_id)' ]; then \
            echo \"SUCCESS|$gpu_id|\$rc|\$(date +%s)|$log_file\" > '$status_file'; \
        else \
            echo \"FAILED|$gpu_id|\$rc|\$(date +%s)|$log_file\" > '$status_file'; \
        fi" C-m 2>/dev/null
}

launch_task() {
    local suite=$1 task_id=$2 gpu_id=$3 pane_info=$4
    record_task_gpu_mapping "$suite" "$task_id" "$gpu_id"
    local new_load
    new_load=$(increment_gpu_load "$gpu_id")
    # Record launch time so cleanup can grant a startup grace period (python/gcc/model
    # loading takes ~60s; pgrep would falsely report "process gone" during this window).
    date +%s > "$TASK_STATUS_DIR/${suite}_task${task_id}.start"
    echo "[$(date '+%H:%M:%S')] Assigned: $suite task_id=$task_id -> GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
    launch_task_on_pane "$suite" "$task_id" "$gpu_id" "$pane_info"
}

# Reconcile GPU load by checking which running tasks have finished.
# Releases GPU capacity for completed/failed tasks.
cleanup_completed_tasks() {
    CLEANED_COUNT=0
    NEW_FAILURE_COUNT=0
    if [ ! -f "$TASK_GPU_MAP_FILE" ] || [ ! -s "$TASK_GPU_MAP_FILE" ]; then
        return 0
    fi
    local temp_map="$TASK_GPU_MAP_FILE.cleanup"
    > "$temp_map"
    while IFS=: read -r task_info gpu_id; do
        [ -z "$task_info" ] && continue
        local suite task_id
        suite=$(echo "$task_info" | cut -d, -f1)
        task_id=$(echo "$task_info" | cut -d, -f2)
        [ -z "$suite" ] || [ -z "$task_id" ] && continue
        local status_file="$TASK_STATUS_DIR/${suite}_task${task_id}.status"
        local start_file="$TASK_STATUS_DIR/${suite}_task${task_id}.start"
        local result_file
        result_file=$(result_file_for "$suite" "$task_id")
        # Result JSON exists -> success
        if [ -f "$result_file" ]; then
            local new_load
            new_load=$(decrement_gpu_load "$gpu_id")
            local done_pane
            done_pane=$(get_task_pane "$suite" "$task_id")
            [ -n "$done_pane" ] && push_free_pane "$done_pane"
            remove_task_pane "$suite" "$task_id"
            rm -f "$status_file" "$start_file"
            CLEANED_COUNT=$((CLEANED_COUNT+1))
            echo "[$(date '+%H:%M:%S')] Completed: $suite task_id=$task_id GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
            continue
        fi
        # Status file exists with FAILED -> reclaim
        if [ -f "$status_file" ]; then
            local status status_gpu status_rc status_ts status_log
            IFS='|' read -r status status_gpu status_rc status_ts status_log < "$status_file"
            if [ "$status" = "FAILED" ]; then
                local new_load
                new_load=$(decrement_gpu_load "$gpu_id")
                local done_pane
                done_pane=$(get_task_pane "$suite" "$task_id")
                [ -n "$done_pane" ] && push_free_pane "$done_pane"
                remove_task_pane "$suite" "$task_id"
                mark_task_failed "$suite" "$task_id" "$gpu_id" "${status_rc:-unknown}" "${status_log:-unknown}"
                NEW_FAILURE_COUNT=$((NEW_FAILURE_COUNT+1))
                echo "[$(date '+%H:%M:%S')] FAILED: $suite task_id=$task_id rc=$status_rc GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
                rm -f "$status_file" "$start_file"
                continue
            fi
            if [ "$status" = "SUCCESS" ]; then
                local new_load
                new_load=$(decrement_gpu_load "$gpu_id")
                local done_pane
                done_pane=$(get_task_pane "$suite" "$task_id")
                [ -n "$done_pane" ] && push_free_pane "$done_pane"
                remove_task_pane "$suite" "$task_id"
                rm -f "$status_file" "$start_file"
                CLEANED_COUNT=$((CLEANED_COUNT+1))
                continue
            fi
        fi
        # Fallback: no result_file and no status_file. Check if the python process for
        # this (suite, task_id) is still alive. If the pane crashed (OOM/SIGKILL) without
        # writing a status file, the process is gone -> treat as failed and release GPU.
        # (remain-on-exit on should prevent this, but this is a safety net.)
        # Grant a startup grace period: python + gcc compile + model loading takes ~60s,
        # during which pgrep may not yet see the process. Skip the check if launched <90s ago.
        # (start_file already defined above alongside status_file)
        local skip_pgrep=0
        if [ -f "$start_file" ]; then
            local launch_ts now_ts
            launch_ts=$(cat "$start_file" 2>/dev/null || echo 0)
            now_ts=$(date +%s)
            if [ $((now_ts - launch_ts)) -lt 90 ]; then
                skip_pgrep=1
            fi
        fi
        if [ "$skip_pgrep" -eq 0 ] && ! pgrep -f "task_suite_name=$suite EVALUATION.task_id=$task_id" >/dev/null 2>&1; then
            local new_load
            new_load=$(decrement_gpu_load "$gpu_id")
            local done_pane
            done_pane=$(get_task_pane "$suite" "$task_id")
            [ -n "$done_pane" ] && push_free_pane "$done_pane"
            remove_task_pane "$suite" "$task_id"
            mark_task_failed "$suite" "$task_id" "$gpu_id" "no_process" "${TASK_LOG_DIR}/${suite}_task${task_id}_gpu${gpu_id}.log"
            NEW_FAILURE_COUNT=$((NEW_FAILURE_COUNT+1))
            echo "[$(date '+%H:%M:%S')] FAILED (process gone, no status): $suite task_id=$task_id GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
            rm -f "$status_file" "$start_file"
            continue
        fi
        # Still running: keep mapping
        echo "$task_info:$gpu_id" >> "$temp_map"
    done < "$TASK_GPU_MAP_FILE"
    mv "$temp_map" "$TASK_GPU_MAP_FILE"
}

# --- tmux session + pane grid ---
# Grid layout: GRID_ROWS x GRID_COLS panes per window. Total concurrent panes
# = NUM_GPUS * MAX_TASKS_PER_GPU. We use 1 window with enough panes.
TOTAL_PANES=$(( NUM_GPUS * MAX_TASKS_PER_GPU ))
GRID_COLS=$MAX_TASKS_PER_GPU
GRID_ROWS=$(( (TOTAL_PANES + GRID_COLS - 1) / GRID_COLS ))
if [ $GRID_ROWS -lt 1 ]; then GRID_ROWS=1; fi

create_grid_layout() {
    # Create TOTAL_PANES panes in window 0. Use -P -F '#{pane_id}' to capture each pane's
    # unique %id (e.g. %1276) at creation time. The %id is required for send-keys — the
    # "session:0.index" form is unreliable in some tmux versions (send-keys silently fails).
    # The first pane (created by new-session) is captured separately.
    PANE_IDS=()
    local first_id
    first_id=$(tmux list-panes -t "$SESSION_NAME:0" -F '#{pane_id}' 2>/dev/null | head -1)
    PANE_IDS+=("$first_id")
    local existing
    existing=$(tmux list-panes -t "$SESSION_NAME:0" 2>/dev/null | wc -l)
    for ((i=existing; i<TOTAL_PANES; i++)); do
        local new_id
        new_id=$(tmux split-window -t "$SESSION_NAME:0" -P -F '#{pane_id}' 2>/dev/null)
        PANE_IDS+=("$new_id")
        tmux select-layout -t "$SESSION_NAME:0" tiled 2>/dev/null
    done
    # remain-on-exit: keep panes alive after their command string finishes, so the shell
    # stays put and the next send-keys reuses it. (The launch command is a multi-statement
    # string: python ...; rc=$?; echo STATUS > file. After it finishes the shell idles.)
    tmux set-option -t "$SESSION_NAME:0" remain-on-exit on 2>/dev/null
    # Give each pane's shell a moment to be ready for send-keys (newly split panes need
    # ~1-2s or the first send-keys is lost, leaving the task unlaunched -> deadlock).
    sleep 2
}

# Kill any stale session, create a fresh detached one
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
    echo "Killed stale tmux session '$SESSION_NAME'"
fi
tmux new-session -d -s "$SESSION_NAME"
create_grid_layout

# Cleanup handler: Ctrl+C (SIGINT) / SIGTERM / exit must tear down the detached tmux
# session, otherwise the python eval processes spawned in its panes keep running as
# orphans after this scheduler script dies. NOTE: `tmux kill-session` alone is NOT
# sufficient — it kills the pane shells but their child `python eval_...` processes
# often survive (verified). So we (1) kill the session, then (2) pkill any residual
# eval processes by command-line match. Guarded by a flag so the normal exit path
# (which also calls cleanup_on_exit) doesn't double-fire.
_CLEANUP_DONE=0
cleanup_on_exit() {
    [ "$_CLEANUP_DONE" = "1" ] && return 0
    _CLEANUP_DONE=1
    echo ""
    echo "[$(date '+%H:%M:%S')] Scheduler exiting — killing tmux session '$SESSION_NAME' and all eval tasks..."
    # (1) Kill the detached tmux session + its pane shells.
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    # (2) Safety net: kill residual python eval processes we spawned. Match the eval
    # script name + THIS run's OUTPUT_DIR (unique per run) so we never kill other runs.
    # Send SIGTERM first (lets python release GPU/handles), then SIGKILL any survivors.
    pkill -f "$EVAL_SCRIPT.*EVALUATION.output_dir=$OUTPUT_DIR" 2>/dev/null || true
    sleep 1
    pkill -9 -f "$EVAL_SCRIPT.*EVALUATION.output_dir=$OUTPUT_DIR" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Cleanup complete."
}
trap cleanup_on_exit INT TERM EXIT

# Free-pane stack: stores tmux %ids (e.g. %1276) of idle panes. Initially all panes are free.
# A task pops a %id when it launches; cleanup pushes the %id back when the task finishes.
FREE_PANES_FILE="$OUTPUT_DIR/free_panes.txt"
: > "$FREE_PANES_FILE"
for pid in "${PANE_IDS[@]}"; do
    echo "$pid" >> "$FREE_PANES_FILE"
done
PANE_LOCK="$OUTPUT_DIR/pane.lock"
touch "$PANE_LOCK"

# Pop a free pane id (atomic). Returns empty string if none available.
pop_free_pane() {
    local pane_id
    pane_id=$(flock "$PANE_LOCK" bash -c '
        if [ -s "$1" ]; then
            head -n1 "$1"
            sed -i "1d" "$1"
        fi
    ' _ "$FREE_PANES_FILE" 2>/dev/null)
    echo "$pane_id"
}

# Push a pane id back to the free stack (atomic).
push_free_pane() {
    local pane_id=$1
    flock "$PANE_LOCK" bash -c 'echo "$2" >> "$1"' _ "$FREE_PANES_FILE" "$pane_id" 2>/dev/null
}

# Map task -> pane_id (so cleanup can return the pane when the task finishes)
TASK_PANE_FILE="$OUTPUT_DIR/task_pane_map.txt"
: > "$TASK_PANE_FILE"
record_task_pane() {
    local suite=$1 task_id=$2 pane_id=$3
    flock "$PANE_LOCK" bash -c 'echo "$2,$3,$4" >> "$1"' _ "$TASK_PANE_FILE" "$suite" "$task_id" "$pane_id" 2>/dev/null
}
get_task_pane() {
    local suite=$1 task_id=$2
    local mapping
    mapping=$(flock "$PANE_LOCK" bash -c 'grep "^$2,$3," "$1" | head -1 | cut -d, -f3' _ "$TASK_PANE_FILE" "$suite" "$task_id" 2>/dev/null)
    echo "${mapping:-}"
}
remove_task_pane() {
    local suite=$1 task_id=$2
    flock "$PANE_LOCK" bash -c 'grep -v "^$2,$3," "$1" > "$1.tmp" && mv "$1.tmp" "$1"' _ "$TASK_PANE_FILE" "$suite" "$task_id" 2>/dev/null
}

init_gpu_load_tracking

# --- Build task array ---
task_array=()
while IFS=, read -r suite task_id; do
    [ -z "$suite" ] && continue
    task_array+=("$suite,$task_id")
done < "$TASK_FILE"
echo "[$(date '+%H:%M:%S')] Loaded ${#task_array[@]} tasks. Starting dynamic scheduling (max $MAX_TASKS_PER_GPU tasks/GPU)..."

# Pending queue (copy of task list)
PENDING_TASKS_FILE="$OUTPUT_DIR/pending_tasks.txt"
cp "$TASK_FILE" "$PENDING_TASKS_FILE"

monitoring_interval=${MONITORING_INTERVAL:-10}
status_interval=${STATUS_INTERVAL:-30}
last_status_time=0

# Initial launch: fill all GPUs up to MAX_TASKS_PER_GPU
max_initial_tasks=$(( NUM_GPUS * MAX_TASKS_PER_GPU ))
initial_launched=0
for task_info in "${task_array[@]}"; do
    [ $initial_launched -ge $max_initial_tasks ] && break
    suite=$(echo "$task_info" | cut -d, -f1)
    task_id=$(echo "$task_info" | cut -d, -f2)
    gpu_id=$(find_least_loaded_gpu)
    [ -z "$gpu_id" ] && break
    pane_id=$(pop_free_pane)
    [ -z "$pane_id" ] && break
    launch_task "$suite" "$task_id" "$gpu_id" "$pane_id"
    record_task_pane "$suite" "$task_id" "$pane_id"
    initial_launched=$((initial_launched+1))
    # Remove from pending
    grep -v "^$suite,$task_id$" "$PENDING_TASKS_FILE" > "$PENDING_TASKS_FILE.tmp" 2>/dev/null || true
    mv "$PENDING_TASKS_FILE.tmp" "$PENDING_TASKS_FILE"
    sleep 0.5
done
echo "[$(date '+%H:%M:%S')] Initial launch: $initial_launched tasks started"

# Main scheduling loop
while true; do
    current_time=$(date +%s)
    cleanup_completed_tasks
    cleaned=$CLEANED_COUNT
    new_failures=$NEW_FAILURE_COUNT
    total_failed=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)

    if [ "$new_failures" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] Detected failed subtask(s), stopping scheduler."
        echo "Failed tasks:"
        cat "$FAILED_TASKS_FILE"
        # Kill remaining tmux panes
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null
        exit 2
    fi

    # All done?
    total_completed=$(find "$OUTPUT_DIR" -type f -name "task*_results.json" | wc -l)
    if [ "$total_completed" -eq "$TOTAL_TASKS" ]; then
        echo "[$(date '+%H:%M:%S')] All $total_completed/$TOTAL_TASKS tasks complete!"
        break
    fi

    # Launch new tasks on freed GPUs
    launched_this_round=0
    temp_pending="$PENDING_TASKS_FILE.processing"
    cp "$PENDING_TASKS_FILE" "$temp_pending" 2>/dev/null || continue
    > "$PENDING_TASKS_FILE"
    while IFS=, read -r suite task_id; do
        [ -z "$suite" ] && continue
        # Already completed?
        if [ -f "$(result_file_for "$suite" "$task_id")" ]; then
            continue
        fi
        # Already running?
        running_gpu=$(get_task_gpu "$suite" "$task_id")
        [ -n "$running_gpu" ] && continue
        # Launch on least-loaded GPU
        gpu_id=$(find_least_loaded_gpu)
        if [ -n "$gpu_id" ]; then
            pane_id=$(pop_free_pane)
            if [ -z "$pane_id" ]; then
                # No free pane (shouldn't happen if GPU load tracked correctly, but guard anyway)
                echo "$suite,$task_id" >> "$PENDING_TASKS_FILE"
                continue
            fi
            launch_task "$suite" "$task_id" "$gpu_id" "$pane_id"
            record_task_pane "$suite" "$task_id" "$pane_id"
            launched_this_round=$((launched_this_round+1))
        else
            # All GPUs full: keep pending
            echo "$suite,$task_id" >> "$PENDING_TASKS_FILE"
        fi
    done < "$temp_pending"
    rm -f "$temp_pending"

    running_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
    pending_count=$(wc -l < "$PENDING_TASKS_FILE" 2>/dev/null || echo 0)
    if [ "$running_count" -eq 0 ] && [ "$pending_count" -eq 0 ] && [ "$total_completed" -lt "$TOTAL_TASKS" ]; then
        echo "[$(date '+%H:%M:%S')] Scheduling inconsistency: nothing running/pending but only $total_completed/$TOTAL_TASKS done."
        [ -s "$FAILED_TASKS_FILE" ] && cat "$FAILED_TASKS_FILE"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null
        exit 2
    fi

    # Periodic status
    if [ $((current_time - last_status_time)) -ge $status_interval ]; then
        echo "[$(date '+%H:%M:%S')] === Status: done $total_completed/$TOTAL_TASKS, running $running_count, pending $pending_count, failed $total_failed ==="
        for gpu in "${GPU_ARRAY[@]}"; do
            load=$(get_gpu_load $gpu)
            echo "  GPU $gpu: $load/$MAX_TASKS_PER_GPU"
        done
        last_status_time=$current_time
    fi
    sleep $monitoring_interval
done

# Cleanup (normal exit path; the EXIT trap also calls this — guarded by _CLEANUP_DONE)
cleanup_on_exit
rm -f "$PENDING_TASKS_FILE" "$PENDING_TASKS_FILE.processing"

echo ""
echo "=========================================="
echo "All tasks finished. Summarizing..."
echo "=========================================="

# Summarize results across all suites (transposed table)
python - "$OUTPUT_DIR" <<'PY'
import json, glob, os, sys
output_dir = sys.argv[1]

# Aggregate per-suite successes/trials from task*_results.json
suite_stats = {}  # suite -> {successes, trials, tasks, duration}
for f in sorted(glob.glob(os.path.join(output_dir, "*", "task*_results.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    suite = d.get("task_suite", "?")
    s = suite_stats.setdefault(suite, {"successes": 0, "trials": 0, "tasks": 0, "duration": 0.0})
    s["successes"] += int(d.get("successes", 0))
    s["trials"] += int(d.get("total_episodes", 0))
    s["tasks"] += 1
    s["duration"] += float(d.get("duration", 0.0))

if not suite_stats:
    print("No result JSONs found. Check task logs in:", output_dir)
    sys.exit(0)

# Fixed suite order (libero_spatial, libero_object, libero_goal, libero_10)
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
ordered = [s for s in SUITE_ORDER if s in suite_stats]
ordered += [s for s in sorted(suite_stats) if s not in ordered]

# Per-suite success rate (successes / trials * 100)
suite_rates = []
suite_times = []
for s in ordered:
    st = suite_stats[s]
    rate = (100.0 * st["successes"] / st["trials"]) if st["trials"] else 0.0
    suite_rates.append(rate)
    suite_times.append(st["duration"] / st["tasks"] if st["tasks"] else 0.0)

# Overall = arithmetic mean of per-suite rates 
overall = sum(suite_rates) / len(suite_rates) if suite_rates else 0.0
overall_time = sum(suite_times) / len(suite_times) if suite_times else 0.0

# Transposed table: suites as columns, metrics as rows, no labels/separators
try:
    import pandas as pd
    df = pd.DataFrame({
        "Task Suite": ordered + ["Overall"],
        "Success Rate (%)": [f"{r:.2f}" for r in suite_rates] + [f"{overall:.2f}"],
    })
    df = df.set_index("Task Suite").T
    print()
    print(df.to_string(index=False))
    # Save summary.csv (transposed format)
    df.to_csv(os.path.join(output_dir, "summary.csv"))
except Exception:
    # Fallback without pandas: manual right-aligned columns
    cols = ordered + ["Overall"]
    vals = [f"{r:.2f}" for r in suite_rates] + [f"{overall:.2f}"]
    w = max(len(c) for c in cols)
    print()
    print("  ".join(c.rjust(w) for c in cols))
    print("  ".join(v.rjust(w) for v in vals))
    with open(os.path.join(output_dir, "summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join(vals) + "\n")

print(f"\nResults dir: {output_dir}")
print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")
PY
