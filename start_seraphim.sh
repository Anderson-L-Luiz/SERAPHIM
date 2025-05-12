#!/bin/bash
SERAPHIM_DIR_START="/home/aimotion_api/SERAPHIM"
CONDA_ENV_NAME_START="seraphim_vllm_env"
CONDA_BASE_PATH_START="/home/aimotion_api/anaconda3"
BACKEND_SCRIPT_START="seraphim_backend.py"
BACKEND_PORT_START=8870
FRONTEND_PORT_START=8869
MODELS_FILE_START="$SERAPHIM_DIR_START/models.txt"

BACKEND_LOG_FILE="$SERAPHIM_DIR_START/seraphim_backend.log"
FRONTEND_LOG_FILE="$SERAPHIM_DIR_START/seraphim_frontend.log"
BACKEND_PID_FILE="$SERAPHIM_DIR_START/seraphim_backend.pid"
FRONTEND_PID_FILE="$SERAPHIM_DIR_START/seraphim_frontend.pid"

is_port_in_use() {
    local port=$1
    if command -v ss > /dev/null; then
        ss -tulnp | grep -q ":${port}\s"
    elif command -v netstat > /dev/null; then
        netstat -tulnp | grep -q ":${port}\s"
    else
        echo "Warning: Neither 'ss' nor 'netstat' found. Cannot check if port $port is in use."
        return 0 
    fi
}

echo "Starting SERAPHIM Application..."
echo "================================="

if [ ! -f "$MODELS_FILE_START" ]; then
    echo "❌ Error: Models file ($MODELS_FILE_START) not found!"
    echo "Please create this file in the SERAPHIM directory ($SERAPHIM_DIR_START) and populate it with your models."
    echo "Format: model_id,Display Name (one model per line)"
    exit 1
elif ! grep -q '[^[:space:]]' "$MODELS_FILE_START"; then 
    echo "⚠️ Warning: Models file ($MODELS_FILE_START) is empty or contains only whitespace."
    echo "The model dropdown in the UI will be empty. Please populate the file with models."
fi


if [ -f "$BACKEND_PID_FILE" ] && ps -p $(cat "$BACKEND_PID_FILE") > /dev/null; then
    echo "❌ Backend already running (PID: $(cat "$BACKEND_PID_FILE")). Use ./stop_seraphim.sh to stop it first."
    exit 1
fi
if is_port_in_use "$BACKEND_PORT_START"; then
    echo "❌ Error: Backend port $BACKEND_PORT_START is already in use. Please free it or change BACKEND_PORT in install.sh and re-run install."
    exit 1
fi

if [ -f "$FRONTEND_PID_FILE" ] && ps -p $(cat "$FRONTEND_PID_FILE") > /dev/null; then
    echo "❌ Frontend server already running (PID: $(cat "$FRONTEND_PID_FILE")). Use ./stop_seraphim.sh to stop it first."
    exit 1
fi
if is_port_in_use "$FRONTEND_PORT_START"; then
    echo "❌ Error: Frontend port $FRONTEND_PORT_START is already in use. Please free it or change FRONTEND_PORT in install.sh and re-run install."
    exit 1
fi

cd "$SERAPHIM_DIR_START" || { echo "Error: Could not navigate to $SERAPHIM_DIR_START"; exit 1; }
echo "Activating Conda environment: $CONDA_ENV_NAME_START..."
_CONDA_SH_PATH="$CONDA_BASE_PATH_START/etc/profile.d/conda.sh"
if [ -z "$CONDA_BASE_PATH_START" ] || [ ! -f "$_CONDA_SH_PATH" ]; then 
    _FALLBACK_CONDA_BASE_PATH=$(conda info --base 2>/dev/null) 
    if [ -n "$_FALLBACK_CONDA_BASE_PATH" ]; then 
        _CONDA_SH_PATH="$_FALLBACK_CONDA_BASE_PATH/etc/profile.d/conda.sh"
        echo "Using fallback Conda base path: $_FALLBACK_CONDA_BASE_PATH"
    fi
fi
if [ ! -f "$_CONDA_SH_PATH" ]; then echo "Error: conda.sh not found. Cannot activate Conda environment."; exit 1; fi

# shellcheck source=/dev/null
. "$_CONDA_SH_PATH"; conda activate "$CONDA_ENV_NAME_START"
if [[ "$CONDA_DEFAULT_ENV" != *"$CONDA_ENV_NAME_START"* ]]; then 
    echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME_START'."
    echo "Current env: $CONDA_DEFAULT_ENV. Conda prefix: $CONDA_PREFIX."
    exit 1
fi
echo "Conda environment '$CONDA_ENV_NAME_START' activated. Path: $CONDA_PREFIX";

echo "Starting Backend Server (port $BACKEND_PORT_START)... Logging to: $BACKEND_LOG_FILE"
nohup python "$BACKEND_SCRIPT_START" >> "$BACKEND_LOG_FILE" 2>&1 &
_BACKEND_PID=$!; echo $_BACKEND_PID > "$BACKEND_PID_FILE"
echo "Backend PID: $_BACKEND_PID."
sleep 3 
if ! ps -p $_BACKEND_PID > /dev/null; then 
    echo "❌ Error: Backend server failed to start. Check $BACKEND_LOG_FILE for details."
    rm -f "$BACKEND_PID_FILE"
    exit 1
fi

echo "Starting Frontend HTTP Server (port $FRONTEND_PORT_START)... Logging to: $FRONTEND_LOG_FILE"
nohup python -m http.server --bind 0.0.0.0 "$FRONTEND_PORT_START" >> "$FRONTEND_LOG_FILE" 2>&1 &
_FRONTEND_PID=$!; echo $_FRONTEND_PID > "$FRONTEND_PID_FILE"
echo "Frontend PID: $_FRONTEND_PID."
sleep 1 
if ! ps -p $_FRONTEND_PID > /dev/null; then 
    echo "❌ Error: Frontend HTTP server failed to start. Check $FRONTEND_LOG_FILE for details."
    if ps -p $_BACKEND_PID > /dev/null; then kill $_BACKEND_PID; fi
    rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
    exit 1
fi

_SERVER_IP_GUESS=$(hostname -I | awk '{print $1}' || echo "YOUR_SERVER_IP_ADDRESS")
echo "================================="
echo "✅ SERAPHIM Application Started!"
echo "   Access Frontend UI at: http://${_SERVER_IP_GUESS}:$FRONTEND_PORT_START"
echo "   (If the IP is incorrect, use your server's actual IP address)"
echo ""
echo "   To stop the application, run: ./stop_seraphim.sh"
echo "================================="
