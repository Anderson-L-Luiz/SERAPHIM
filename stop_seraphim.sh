#!/bin/bash
SERAPHIM_DIR_STOP="/home/aimotion_api/SERAPHIM"
BACKEND_PID_FILE_STOP="$SERAPHIM_DIR_STOP/seraphim_backend.pid"
FRONTEND_PID_FILE_STOP="$SERAPHIM_DIR_STOP/seraphim_frontend.pid"
echo "Stopping SERAPHIM Application..."
stop_process() {
    local pid_file="$1"; local process_name="$2"
    if [ -f "$pid_file" ]; then
        _PID_TO_KILL=$(cat "$pid_file")
        if ps -p "$_PID_TO_KILL" > /dev/null; then
            echo "Stopping $process_name (PID: $_PID_TO_KILL)..."; kill "$_PID_TO_KILL"; sleep 1
            if ps -p "$_PID_TO_KILL" > /dev/null; then kill -9 "$_PID_TO_KILL"; sleep 1; fi
            if ps -p "$_PID_TO_KILL" > /dev/null; then echo "❌ Error stopping $process_name."; else echo "✅ $process_name stopped."; fi
        else echo "ℹ️ $process_name (PID $_PID_TO_KILL) not running."; fi
        rm -f "$pid_file"
    else echo "⚠️ $process_name PID file not found."; fi
}
stop_process "$BACKEND_PID_FILE_STOP" "Backend Server"
stop_process "$FRONTEND_PID_FILE_STOP" "Frontend Server"
echo "SERAPHIM Stop Attempted."
