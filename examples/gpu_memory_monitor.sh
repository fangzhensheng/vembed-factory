#!/bin/bash
# GPU Memory Monitor
#
# This script monitors GPU memory usage and enforces a limit.
# If memory usage exceeds the limit, the monitored process is killed.
#
# Usage:
#   source gpu_memory_monitor.sh
#   start_gpu_monitor <gpu_id> <memory_limit_gb> <process_pid>
#
# Example:
#   start_gpu_monitor 7 24 $TRAINING_PID

start_gpu_monitor() {
    local gpu_id=$1
    local limit_gb=$2
    local training_pid=$3

    if [ -z "$gpu_id" ] || [ -z "$limit_gb" ] || [ -z "$training_pid" ]; then
        echo "ERROR: start_gpu_monitor requires 3 arguments: <gpu_id> <limit_gb> <training_pid>"
        return 1
    fi

    local limit_mb=$((limit_gb * 1024))

    echo "[GPU Monitor] Starting GPU memory monitor for GPU $gpu_id (limit: ${limit_gb}GB, PID: $training_pid)..."

    while kill -0 $training_pid 2>/dev/null; do
        # Get GPU memory usage for specific GPU
        local mem_used=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')

        if [ -z "$mem_used" ]; then
            sleep 1
            continue
        fi

        # Check if memory exceeds limit
        if [ "$mem_used" -gt "$limit_mb" ]; then
            echo "[GPU Monitor] ERROR: GPU $gpu_id memory usage (${mem_used}MB) exceeded limit (${limit_mb}MB)"
            echo "[GPU Monitor] Killing process and all children..."

            # Kill the main process
            kill -9 $training_pid 2>/dev/null || true

            # Kill all child processes (workers in accelerate/FSDP)
            local child_pids=$(pgrep -P $training_pid 2>/dev/null)
            if [ -n "$child_pids" ]; then
                echo "[GPU Monitor] Killing child processes: $child_pids"
                echo "$child_pids" | xargs kill -9 2>/dev/null || true
            fi

            sleep 2
            return 1
        fi

        # Log memory usage every 30 seconds (only if close to limit)
        local percentage=$((mem_used * 100 / limit_mb))
        if [ $((percentage % 10)) -eq 0 ] 2>/dev/null; then
            if [ "$percentage" -gt 80 ]; then
                echo "[GPU Monitor] GPU $gpu_id: ${mem_used}MB / ${limit_mb}MB (${percentage}%)"
            fi
        fi

        sleep 2  # Check every 2 seconds
    done
}

# Export function for subshells
export -f start_gpu_monitor
