#!/bin/bash

# TensorBoard configuration
LOG_DIR="/home/ubuntu/subbu_ws/src/ViT-Learning/runs/vit_oxford"
PORT=5252

# Restart TensorBoard every 30 minutes
RESTART_INTERVAL=1800

TB_PID=""

cleanup() {
    echo "Stopping TensorBoard..."

    if [ -n "$TB_PID" ] && kill -0 "$TB_PID" 2>/dev/null; then
        kill "$TB_PID"
        wait "$TB_PID" 2>/dev/null
    fi

    exit 0
}

# Handle Ctrl+C / termination
trap cleanup SIGINT SIGTERM

while true
do
    echo "=========================================="
    echo "Starting TensorBoard"
    echo "Time: $(date)"
    echo "Log directory: $LOG_DIR"
    echo "=========================================="

    tensorboard \
        --logdir "$LOG_DIR" \
        --port "$PORT" \
        --bind_all &

    TB_PID=$!

    echo "TensorBoard PID: $TB_PID"

    # Keep TensorBoard alive for configured duration
    sleep "$RESTART_INTERVAL"

    echo ""
    echo "Restart interval reached."
    echo "Stopping TensorBoard PID $TB_PID..."

    if kill -0 "$TB_PID" 2>/dev/null; then
        kill "$TB_PID"
        wait "$TB_PID" 2>/dev/null
    fi

    # Give the OS a moment to release memory/port
    sleep 3

    echo "Restarting TensorBoard..."
done
