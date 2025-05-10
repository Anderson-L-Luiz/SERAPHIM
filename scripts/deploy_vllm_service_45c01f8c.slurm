#!/bin/bash
#SBATCH --job-name=vllm_service
#SBATCH --output=/home/aimotion_api/SERAPHIM/scripts/vllm_service_%j.out
#SBATCH --error=/home/aimotion_api/SERAPHIM/scripts/vllm_service_%j.err
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=NONE

echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output File (template based on %j): /home/aimotion_api/SERAPHIM/scripts/vllm_service_%j.out"
echo "Slurm Error File (template based on %j): /home/aimotion_api/SERAPHIM/scripts/vllm_service_%j.err"
echo "Model: EleutherAI/gpt-j-6b"
echo "Conda Env: seraphim_vllm_env"
echo "=================================================================="

CONDA_BASE_PATH_SLURM="$(conda info --base)"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Could not determine Conda base path."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then 
    echo "Sourcing Conda from $CONDA_SH_PATH"
    . "$CONDA_SH_PATH"
else 
    echo "WARN: conda.sh not found at $CONDA_SH_PATH."
fi

echo "Activating Conda Environment: seraphim_vllm_env"
conda activate "seraphim_vllm_env"
if [[ "$CONDA_PREFIX" != *"seraphim_vllm_env"* ]]; then 
    echo "ERROR: Failed to activate conda environment 'seraphim_vllm_env'. CONDA_PREFIX=$CONDA_PREFIX"; 
    exit 1;
else
    echo "Conda environment 'seraphim_vllm_env' activated. Path: $CONDA_PREFIX";
fi

HF_TOKEN_VALUE=""
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

export VLLM_USE_TRITON_FLASH_ATTN="True" # Or other vLLM specific env vars
export VLLM_CONFIGURE_LOGGING="0" 
export VLLM_NO_USAGE_STATS="True"
export VLLM_DO_NOT_TRACK="True"

VLLM_LOG_FILE_FOR_JOB="/home/aimotion_api/SERAPHIM/scripts/vllm_logs/vllm_service_$SLURM_JOB_ID_vllm_service.log"
mkdir -p "/home/aimotion_api/SERAPHIM/scripts/vllm_logs" # Ensure vLLM log directory exists

echo -e "\nStarting vLLM API Server in background..."
echo "Command: vllm serve "EleutherAI/gpt-j-6b" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384"
echo "vLLM service log will be: $VLLM_LOG_FILE_FOR_JOB"

# Run vLLM in the background using nohup, redirecting stdout/stderr to its specific log file
nohup vllm serve "EleutherAI/gpt-j-6b" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384 > "$VLLM_LOG_FILE_FOR_JOB" 2>&1 &
VLLM_PID=$!
echo "vLLM Service potentially started with PID: $VLLM_PID. Waiting briefly for initialization..."
sleep 15 # Allow time for vLLM to start and log initial messages

if ps -p $VLLM_PID > /dev/null; then 
    echo "vLLM process $VLLM_PID appears to be running."
    echo "Initial logs should be available in $VLLM_LOG_FILE_FOR_JOB"
else 
    echo "ERROR: vLLM process $VLLM_PID does not seem to be running. Check $VLLM_LOG_FILE_FOR_JOB for errors."
fi

echo ""
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SERVICE STATUS ✝"
echo "Job ID: $SLURM_JOB_ID on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Model: EleutherAI/gpt-j-6b"
echo "Configured Service Port: 8000"
echo "vLLM Service Log File: $VLLM_LOG_FILE_FOR_JOB"
echo "---"
echo "The vLLM service is running in the background (PID: $VLLM_PID)."
echo "To find the API endpoint:"
echo "  1. Check the vLLM log: cat $VLLM_LOG_FILE_FOR_JOB"
echo "  2. Look for lines like 'Uvicorn running on http://<ip_address>:8000'"
echo "  3. The API will be accessible on the node $(hostname -s) at that address."
echo "     Typically: http://$(hostname -s):8000"
echo "     OpenAI API Docs (Swagger UI): http://$(hostname -s):8000/docs"
echo "=================================================================="
echo "Slurm script finished its main tasks. The vLLM service (PID: $VLLM_PID) should continue in the background."
