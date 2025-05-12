#!/bin/bash
#SBATCH --job-name=Deepseek_for_Commit_User_Modification
#SBATCH --output=/home/aimotion_api/SERAPHIM/scripts/Deepseek_for_Commit_User_Modification_%j.out
#SBATCH --error=/home/aimotion_api/SERAPHIM/scripts/Deepseek_for_Commit_User_Modification_%j.err
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=NONE
# ... (rest of sbatch script content from previous version, no changes needed here) ...
echo "Current ulimit -n (soft): $(ulimit -Sn)"
echo "Current ulimit -n (hard): $(ulimit -Hn)"
ulimit -n 10240 
if [ $? -eq 0 ]; then echo "Successfully set ulimit -n to $(ulimit -Sn)"; else echo "WARN: Failed to set ulimit -n. Current: $(ulimit -Sn). Check hard limits if issues persist."; fi
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output File: /home/aimotion_api/SERAPHIM/scripts/Deepseek_for_Commit_User_Modification_$SLURM_JOB_ID.out"
echo "Slurm Error File: /home/aimotion_api/SERAPHIM/scripts/Deepseek_for_Commit_User_Modification_$SLURM_JOB_ID.err"
echo "Model: deepseek-ai/deepseek-moe-16b-chat"
echo "Target Service Port: 8000"
echo "Conda Env: seraphim_vllm_env"
echo "Max Model Length Requested: 16384"
echo "vLLM service will run in the FOREGROUND of this Slurm job."
echo "=================================================================="
CONDA_BASE_PATH_SLURM="$(conda info --base)"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Conda base path empty."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then . "$CONDA_SH_PATH"; else echo "WARN: conda.sh not found."; fi
conda activate "seraphim_vllm_env"
if [[ "$CONDA_PREFIX" != *"seraphim_vllm_env"* ]]; then echo "ERROR: Failed to activate conda. Prefix: $CONDA_PREFIX"; exit 1; fi
echo "Conda env 'seraphim_vllm_env' activated. Path: $CONDA_PREFIX";
HF_TOKEN_VALUE=""
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi
export VLLM_CONFIGURE_LOGGING="0"
export VLLM_NO_USAGE_STATS="True"
export VLLM_DO_NOT_TRACK="True"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN set to 1."
echo -e "\nStarting vLLM API Server in FOREGROUND..."
echo "Command: vllm serve "deepseek-ai/deepseek-moe-16b-chat" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384"
echo "vLLM logs will be in Slurm output/error files."
echo "--- vLLM Service Starting (Output will follow) ---"
vllm serve "deepseek-ai/deepseek-moe-16b-chat" \
    --host "0.0.0.0" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384
VLLM_EXIT_CODE=$?
echo "--- vLLM Service Ended (Exit Code: $VLLM_EXIT_CODE) ---"
echo "=================================================================="
echo "✝ SERAPHIM vLLM Job - FINAL STATUS ✝"
if [ $VLLM_EXIT_CODE -eq 0 ]; then echo "vLLM exited cleanly or was terminated."; else echo "ERROR: vLLM exited with code: $VLLM_EXIT_CODE."; fi
echo "Slurm job $SLURM_JOB_ID finished."
echo "=================================================================="
