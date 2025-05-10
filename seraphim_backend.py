# seraphim_backend.py

import os
import subprocess
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

SERAPHIM_DIR_PY = "/home/aimotion_api/SERAPHIM"
# SCRIPTS_DIR_PY is where Slurm scripts are saved AND where their .out/.err files will go.
SCRIPTS_DIR_PY = "/home/aimotion_api/SERAPHIM/scripts" 
# VLLM_LOG_DIR_PY is not directly used for vLLM output in Slurm script anymore.
# It could be used by the backend for its own general logging if desired.
VLLM_LOG_DIR_PY = "/home/aimotion_api/SERAPHIM/scripts/vllm_service_specific_logs" 
CONDA_ENV_NAME_PY = "seraphim_vllm_env"
BACKEND_PORT_PY = 8870

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"],
)

class SlurmConfig(BaseModel):
    selected_model: str; service_port: str; hf_token: str | None = None
    job_name: str; time_limit: str; gpus: str; cpus_per_task: str; mem: str
    mail_user: str | None = None

def generate_sbatch_script_content(config: SlurmConfig, scripts_dir: str, conda_env_name: str) -> tuple[str, str, str, str]:
    # scripts_dir is where .slurm, .out, .err files will reside.
    conda_base_path_for_slurm_script = "$(conda info --base)"
    escaped_selected_model_for_vllm_cmd = config.selected_model.replace('"', '\\"')

    vllm_serve_command = f'vllm serve "{escaped_selected_model_for_vllm_cmd}"'
    model_args = [
        f'--host "0.0.0.0"', f'--port {config.service_port}', '--trust-remote-code'
    ]
    max_model_len = 16384 # Default, user example used 4096 for Llama-2-7b
    # Adjust max_model_len based on model, similar to user's example logic for Llama-2
    if "llama-2-7b" in config.selected_model.lower() or "llama2-7b" in config.selected_model.lower():
        max_model_len = 4096 
        logger.info(f"Adjusted max_model_len to {max_model_len} for {config.selected_model}")
    elif "mixtral" in config.selected_model.lower():
        max_model_len = 32768
        logger.info(f"Adjusted max_model_len to {max_model_len} for {config.selected_model}")
    
    # Add Pixtral specific args from original frontend logic
    if "pixtral" in config.selected_model.lower():
        model_args.append('--guided-decoding-backend=lm-format-enforcer')
        model_args.append("--limit_mm_per_prompt 'image=8'")
        if "mistralai/Pixtral-12B-2409" in config.selected_model:
             model_args.extend([
                 '--enable-auto-tool-choice', '--tool-call-parser=mistral',
                 '--tokenizer_mode mistral', '--revision aaef4baf771761a81ba89465a18e4427f3a105f9'
             ])
    model_args.append(f'--max-model-len {max_model_len}')
    vllm_serve_command_full = vllm_serve_command + " \\\n    " + " \\\n    ".join(model_args)

    mail_type_line = f"#SBATCH --mail-type=ALL\\n#SBATCH --mail-user={config.mail_user}" if config.mail_user else "#SBATCH --mail-type=NONE"
    safe_filename_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config.job_name)
    unique_id = str(uuid.uuid4())[:8]
    script_filename = f"deploy_{safe_filename_job_name}_{unique_id}.slurm"
    script_path = os.path.join(scripts_dir, script_filename)

    # Slurm output/error files will be in SCRIPTS_DIR_PY (scripts_dir)
    slurm_out_file = os.path.join(scripts_dir, f"{safe_filename_job_name}_%j.out") # %j is Slurm's job ID
    slurm_err_file = os.path.join(scripts_dir, f"{safe_filename_job_name}_%j.err")

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={config.job_name}
#SBATCH --output={slurm_out_file}
#SBATCH --error={slurm_err_file}
#SBATCH --time={config.time_limit}
#SBATCH --gres=gpu:{config.gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem}
{mail_type_line}

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
echo "Slurm Output File: {slurm_out_file.replace('%j', '$SLURM_JOB_ID')}"
echo "Slurm Error File: {slurm_err_file.replace('%j', '$SLURM_JOB_ID')}"
echo "Model: {config.selected_model}"
echo "Target Service Port: {config.service_port}"
echo "Conda Env: {conda_env_name}"
echo "Max Model Length: {max_model_len}"
echo "vLLM service will run in the FOREGROUND of this Slurm job."
echo "=================================================================="

CONDA_BASE_PATH_SLURM="{conda_base_path_for_slurm_script}"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Could not determine Conda base path."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then . "$CONDA_SH_PATH"; else echo "WARN: conda.sh not found."; fi

conda activate "{conda_env_name}"
if [[ "$CONDA_PREFIX" != *"{conda_env_name}"* ]]; then 
    echo "ERROR: Failed to activate conda env '{conda_env_name}'. CONDA_PREFIX=$CONDA_PREFIX"; exit 1;
fi
echo "Conda env '{conda_env_name}' activated. Path: $CONDA_PREFIX";

HF_TOKEN_VALUE="{config.hf_token or ''}"
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

export VLLM_CONFIGURE_LOGGING="0"; export VLLM_NO_USAGE_STATS="True"; export VLLM_DO_NOT_TRACK="True"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" # Enable if absolutely necessary and VRAM allows

echo -e "\\nStarting vLLM API Server in the FOREGROUND..."
echo "Command: {vllm_serve_command_full}"
echo "vLLM service output/errors will be in Slurm output/error files specified above."
echo "--- vLLM Service Starting (Output will follow) ---"

# Execute vLLM Server in the FOREGROUND
{vllm_serve_command_full}

VLLM_EXIT_CODE=$?
echo "--- vLLM Service Ended (Exit Code: $VLLM_EXIT_CODE) ---"
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - FINAL STATUS ✝"
if [ $VLLM_EXIT_CODE -eq 0 ]; then
    echo "vLLM service exited cleanly or was terminated."
else
    echo "ERROR: vLLM service exited with error code: $VLLM_EXIT_CODE."
    echo "Please check Slurm error file: {slurm_err_file.replace('%j', '$SLURM_JOB_ID')}"
fi
echo "Slurm job $SLURM_JOB_ID finished."
echo "=================================================================="
"""
    return script_path, sbatch_content, slurm_out_file.replace('%j', '$SLURM_JOB_ID'), slurm_err_file.replace('%j', '$SLURM_JOB_ID')

@app.post("/api/deploy")
async def deploy_vllm_service_api(config: SlurmConfig, request: Request):
    client_host = request.client.host if request.client else "Unknown"
    logger.info(f"Received deployment request from {client_host} for model: {config.selected_model}")
    try:
        os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
        # os.makedirs(VLLM_LOG_DIR_PY, exist_ok=True) # Less critical now
    except OSError as e:
        logger.error(f"Error creating script directory: {e}")
        raise HTTPException(status_code=500, detail="Server error: Could not create script dir.")

    script_path, sbatch_content, slurm_out_pattern, slurm_err_pattern = generate_sbatch_script_content(
        config, SCRIPTS_DIR_PY, CONDA_ENV_NAME_PY
    )
    try:
        with open(script_path, "w") as f: f.write(sbatch_content)
        os.chmod(script_path, 0o755)
        logger.info(f"Slurm script saved to {script_path}")
    except IOError as e:
        logger.error(f"Error writing/chmod sbatch script {script_path}: {e}")
        raise HTTPException(status_code=500, detail="Server error: Could not write/chmod sbatch script.")

    try:
        submit_command = ["sbatch", script_path]
        logger.info(f"Submitting Slurm job: {' '.join(submit_command)}")
        process = subprocess.run(submit_command, capture_output=True, text=True, check=True, timeout=30)
        job_id_message = process.stdout.strip()
        job_id = job_id_message.split(" ")[-1].strip() if "Submitted batch job" in job_id_message else "Unknown"
        logger.info(f"Sbatch successful. Output: '{job_id_message}', Parsed Job ID: {job_id}")
        
        actual_slurm_out = slurm_out_pattern.replace('$SLURM_JOB_ID', job_id if job_id != "Unknown" else "<JOB_ID>")
        actual_slurm_err = slurm_err_pattern.replace('$SLURM_JOB_ID', job_id if job_id != "Unknown" else "<JOB_ID>")

        return {
            "message": f"Slurm job submitted! ({job_id_message})", "job_id": job_id,
            "script_path": script_path,
            "slurm_output_file_pattern": actual_slurm_out, # User will check this for vLLM logs
            "slurm_error_file_pattern": actual_slurm_err,   # And this for errors
            "monitoring_note": f"vLLM service runs in foreground. Monitor Slurm output ({actual_slurm_out}) for service logs and errors ({actual_slurm_err})."
        }
    except subprocess.TimeoutExpired:
        logger.error("sbatch command timed out.")
        raise HTTPException(status_code=500, detail="Server error: sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        detail_msg = f"Sbatch failed. RC: {e.returncode}. Stderr: {e.stderr.strip()}" if e.stderr.strip() else "Sbatch failed. Check backend logs."
        logger.error(detail_msg)
        raise HTTPException(status_code=500, detail=detail_msg)
    except FileNotFoundError:
        logger.error("'sbatch' not found.")
        raise HTTPException(status_code=500, detail="Server error: 'sbatch' not found.")
    except Exception as e:
        logger.error(f"Unexpected error during sbatch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")

if __name__ == "__main__":
    os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    # os.makedirs(VLLM_LOG_DIR_PY, exist_ok=True) # Less critical now
    logger.info(f"Starting SERAPHIM Backend Server on http://0.0.0.0:{BACKEND_PORT_PY}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT_PY)
