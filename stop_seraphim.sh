#!/bin/bash
SERAPHIM_DIR_STOP="/home/aimotion_api/SERAPHIM"
BACKEND_PID_FILE_STOP="$SERAPHIM_DIR_STOP/seraphim_backend.pid"
FRONTEND_PID_FILE_STOP="$SERAPHIM_DIR_STOP/seraphim_frontend.pid"
echo "Stopping SERAPHIM Application..."
stop_process() {
    local pid_file="$1"; local process_name="$2"
    if [ -f "$pid_file" ]; then
        _PID_TO_KILL=$(cat "$pid_file")
        if [ -n "$_PID_TO_KILL" ] && ps -p "$_PID_TO_KILL" > /dev/null; then
            echo "Stopping $process_name (PID: $_PID_TO_KILL)..."; 
            kill "$_PID_TO_KILL"; 
            for i in {1..5}; do 
                if ! ps -p "$_PID_TO_KILL" > /dev/null; then break; fi; 
                sleep 0.5; 
            done
            if ps -p "$_PID_TO_KILL" > /dev/null; then 
                echo "Force stopping $process_name (PID: $_PID_TO_KILL)...";
                kill -9 "$_PID_TO_KILL"; sleep 0.5; 
            fi
            if ps -p "$_PID_TO_KILL" > /dev/null; then 
                echo "❌ Error: Failed to stop $process_name (PID: $_PID_TO_KILL). Manual check required."; 
            else 
                echo "✅ $process_name stopped."; 
            fi
        else 
            echo "ℹ️ $process_name (PID from file: $_PID_TO_KILL) not running or PID is invalid."
        fi
        rm -f "$pid_file"
    else 
        echo "⚠️ $process_name PID file (${pid_file}) not found. Cannot stop."
    fi
}
stop_process "$BACKEND_PID_FILE_STOP" "Backend Server"
stop_process "$FRONTEND_PID_FILE_STOP" "Frontend Server"
echo "SERAPHIM Stop Process Attempted."
