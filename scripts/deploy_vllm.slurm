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

# Attempt to increase the open files limit.
# This might be restricted by system policies.
echo "Current ulimit -n (soft): $(ulimit -Sn)"
echo "Current ulimit -n (hard): $(ulimit -Hn)"
ulimit -n 10240 # You can try a higher value like 65535 if needed and permitted
if [ $? -eq 0 ]; then
    echo "Successfully set ulimit -n to $(ulimit -Sn)"
else
    echo "WARN: Failed to set ulimit -n. Current value: $(ulimit -Sn). This might lead to 'Too many open files' errors for the service."
fi
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: ${SLURM_JOB_ID} running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output/Error files will be in: /home/aimotion_api/SERAPHIM/scripts/"
echo "Model: meta-llama/Llama-2-7b-chat-hf"
echo "Target Service Port: 8000"
echo "Conda Env: seraphim_vllm_env"
echo "Max Model Length: 4096 (adjusted for model compatibility)"
echo "vLLM service will run in the foreground of this Slurm job."
echo "=================================================================="

# Source Conda
CONDA_BASE_PATH_SLURM="$(conda info --base)"
if [ -z "${CONDA_BASE_PATH_SLURM}" ]; then echo "ERROR: Could not determine Conda base path."; exit 1; fi
CONDA_SH_PATH="${CONDA_BASE_PATH_SLURM}/etc/profile.d/conda.sh"

if [ -f "${CONDA_SH_PATH}" ]; then
    echo "Sourcing Conda from: ${CONDA_SH_PATH}"
    . "${CONDA_SH_PATH}" # Use . to source
else
    echo "WARN: conda.sh not found at ${CONDA_SH_PATH}."
fi

echo "Activating Conda Environment: seraphim_vllm_env"
conda activate "seraphim_vllm_env" # Quoted

# Corrected Conda activation check
if [[ "$CONDA_PREFIX" != *"seraphim_vllm_env"* ]]; then
    echo "ERROR: Failed to activate conda environment 'seraphim_vllm_env'. Actual CONDA_PREFIX was: '$CONDA_PREFIX'"; exit 1;
else
    echo "Conda environment 'seraphim_vllm_env' activated. Path: $CONDA_PREFIX";
fi

# Set HF Token (Usually not required for public Llama-2 models)
if [ -n "" ]; then export HF_TOKEN=""; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

# Set vLLM Env Vars
export VLLM_CONFIGURE_LOGGING="0";
export VLLM_NO_USAGE_STATS="True";
export VLLM_DO_NOT_TRACK="True";
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" # Only if strictly needed and VRAM is sufficient for 16384

echo -e "\nStarting vLLM API Server in the foreground..."
echo "Command to be executed:"
# Updated command for foreground execution
ESCAPED_CMD_FOR_ECHO=$(printf '%s' "vllm serve "meta-llama/Llama-2-7b-chat-hf" --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096" | sed 's/\$/\\\\/g; s/"/\"/g')
echo -e "${ESCAPED_CMD_FOR_ECHO}"
echo "vLLM service output (including API endpoint and any errors) will be in Slurm output/error files:"
echo "Output: /home/aimotion_api/SERAPHIM/scripts/vllm_service_${SLURM_JOB_ID}.out"
echo "Error: /home/aimotion_api/SERAPHIM/scripts/vllm_service_${SLURM_JOB_ID}.err"
echo "--- vLLM Service Starting ---"

# Execute vLLM Server in the FOREGROUND
# Output and errors will go to SBATCH output/error files
vllm serve "meta-llama/Llama-2-7b-chat-hf" --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096

VLLM_EXIT_CODE=$? # Capture the exit code of the vllm serve command

echo "--- vLLM Service Ended ---"
echo ""
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SERVICE STATUS ✝"
echo "Job ID: ${SLURM_JOB_ID} on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Model: meta-llama/Llama-2-7b-chat-hf"
echo "Configured Service Port: 8000"

if [ ${VLLM_EXIT_CODE} -eq 0 ]; then
    echo "vLLM service was running and exited cleanly (or was manually stopped)."
    echo "However, as it ran in the foreground, the job will terminate now."
    echo "If you intended it to run long-term, this foreground mode is mainly for debugging startup."
    echo "To run long-term after debugging, you would reintroduce nohup and &."
else
    echo "ERROR: vLLM service exited with error code: ${VLLM_EXIT_CODE}."
    echo "Please check the Slurm output/error files for detailed messages from vLLM:"
    echo "Output: /home/aimotion_api/SERAPHIM/scripts/vllm_service_${SLURM_JOB_ID}.out"
    echo "Error: /home/aimotion_api/SERAPHIM/scripts/vllm_service_${SLURM_JOB_ID}.err"
fi
echo "=================================================================="

echo "Slurm script has finished."
# The Slurm job will end when vllm serve terminates.
