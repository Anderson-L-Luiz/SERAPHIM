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

echo "Current ulimit -n (soft): $(ulimit -Sn)"
echo "Current ulimit -n (hard): $(ulimit -Hn)"
ulimit -n 10240 # Attempt to increase open files limit
if [ $? -eq 0 ]; then
    echo "Successfully set ulimit -n to $(ulimit -Sn)"
else
    echo "WARN: Failed to set ulimit -n. Current value: $(ulimit -Sn)."
fi
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output File: /home/aimotion_api/SERAPHIM/scripts/vllm_service_$SLURM_JOB_ID.out"
echo "Slurm Error File: /home/aimotion_api/SERAPHIM/scripts/vllm_service_$SLURM_JOB_ID.err"
echo "Model: meta-llama/Llama-2-7b-chat-hf"
echo "Target Service Port: 8000"
echo "Conda Env: seraphim_vllm_env"
echo "Max Model Length: 4096"
echo "vLLM service will run in the FOREGROUND of this Slurm job."
echo "=================================================================="

CONDA_BASE_PATH_SLURM="$(conda info --base)"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Could not determine Conda base path."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then . "$CONDA_SH_PATH"; else echo "WARN: conda.sh not found."; fi

conda activate "seraphim_vllm_env"
if [[ "$CONDA_PREFIX" != *"seraphim_vllm_env"* ]]; then 
    echo "ERROR: Failed to activate conda env 'seraphim_vllm_env'. CONDA_PREFIX=$CONDA_PREFIX"; exit 1;
fi
echo "Conda env 'seraphim_vllm_env' activated. Path: $CONDA_PREFIX";

HF_TOKEN_VALUE=""
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

export VLLM_CONFIGURE_LOGGING="0"; export VLLM_NO_USAGE_STATS="True"; export VLLM_DO_NOT_TRACK="True"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" # Enable if absolutely necessary and VRAM allows

echo -e "\nStarting vLLM API Server in the FOREGROUND..."
echo "Command: vllm serve "meta-llama/Llama-2-7b-chat-hf" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096"
echo "vLLM service output/errors will be in Slurm output/error files specified above."
echo "--- vLLM Service Starting (Output will follow) ---"

# Execute vLLM Server in the FOREGROUND
vllm serve "meta-llama/Llama-2-7b-chat-hf" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096

VLLM_EXIT_CODE=$?
echo "--- vLLM Service Ended (Exit Code: $VLLM_EXIT_CODE) ---"
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - FINAL STATUS ✝"
if [ $VLLM_EXIT_CODE -eq 0 ]; then
    echo "vLLM service exited cleanly or was terminated."
else
    echo "ERROR: vLLM service exited with error code: $VLLM_EXIT_CODE."
    echo "Please check Slurm error file: /home/aimotion_api/SERAPHIM/scripts/vllm_service_$SLURM_JOB_ID.err"
fi
echo "Slurm job $SLURM_JOB_ID finished."
echo "=================================================================="
