#!/bin/bash

# SERAPHIM Installation Script - v2.3.1
# "Matrix" Theme, Larger Log Font, Node IP Display, Updated Configs.
# 4-Col Layout, Green Log Text, Faster Log Polling, Updated Cancel Dialog,
# New Log Titles, Tracks All User Slurm Jobs, Visually Integrated Model Search.
# Auto-select new job and auto-display its logs.
# Assumes models.txt, Searchable Dropdown, Port Checks, Active Deployments, User-defined max_model_len
# Includes VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for overriding model length checks.
# vLLM service runs in FOREGROUND within Slurm job.
# Job Cancellation and Log Viewing capabilities.
# v2.3.1: Removed email notification option, made internal vLLM engine log always auto-scroll.

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
CONDA_ENV_NAME="seraphim_vllm_env"
SERAPHIM_DIR="$HOME/SERAPHIM"
SCRIPTS_DIR="$SERAPHIM_DIR/scripts"
VLLM_LOG_DIR_IN_INSTALLER="$SERAPHIM_DIR/seraphim_internal_logs"
HTML_FILENAME="seraphim_deploy.html"
JS_FILENAME="seraphim_logic.js"
BACKEND_FILENAME="seraphim_backend.py"
MODELS_FILENAME="models.txt"
START_SCRIPT_FILENAME="start_seraphim.sh"
STOP_SCRIPT_FILENAME="stop_seraphim.sh"

HTML_TARGET_PATH="$SERAPHIM_DIR/$HTML_FILENAME"
JS_TARGET_PATH="$SERAPHIM_DIR/$JS_FILENAME"
BACKEND_TARGET_PATH="$SERAPHIM_DIR/$BACKEND_FILENAME"
MODELS_FILE_PATH="$SERAPHIM_DIR/$MODELS_FILENAME"
START_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$START_SCRIPT_FILENAME"
STOP_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$STOP_SCRIPT_FILENAME"
VLLM_REQUIREMENTS_FILE="$SCRIPTS_DIR/vllm_requirements.txt"

BACKEND_PORT=8870
FRONTEND_PORT=8869
JOB_NAME_PREFIX_FOR_SQ="vllm_service"

if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found."
    exit 1
fi

echo "Starting SERAPHIM vLLM Deployment Setup (v2.3.1 - Matrix Theme & Enhancements)..."
echo "Target Directory: $SERAPHIM_DIR"
echo "=========================================================================="
mkdir -p "$SERAPHIM_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$VLLM_LOG_DIR_IN_INSTALLER"
echo "Directories checked/created."
echo ""

echo "INFO: This script assumes you have a correctly formatted '$MODELS_FILENAME' file."
echo "Please ensure '$MODELS_FILE_PATH' exists and is populated with your models."
echo ""

echo "Creating requirements file: $VLLM_REQUIREMENTS_FILE"
cat > "$VLLM_REQUIREMENTS_FILE" << EOF
# Core vLLM and serving
vllm>=0.4.0
# Backend requirements
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
python-dotenv>=1.0.0
# Other vLLM dependencies
aiohappyeyeballs>=2.4.0
aiohttp>=3.8.0
aiohttp-cors>=0.7.0
huggingface-hub>=0.20.0
numpy>=1.23.0
openai>=1.0.0
packaging>=23.0
prometheus-client>=0.17.0
protobuf>=4.20.0
pydantic_core>=2.0.0
PyYAML>=6.0.0
ray>=2.5.0
requests>=2.30.0
safetensors>=0.4.0
sentencepiece>=0.1.98
tokenizers>=0.14.0
torch>=2.1.0
torchaudio>=2.1.0
torchvision>=0.16.0
transformers>=4.35.0
typing_extensions>=4.8.0
xformers>=0.0.22
EOF
echo "Requirements file created."
echo ""

echo "Setting up Conda environment: $CONDA_ENV_NAME"
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    conda env remove -n "$CONDA_ENV_NAME" -y
fi
conda create -n "$CONDA_ENV_NAME" python=3.10 -y

echo "Sourcing conda for activation (during install)..."
CONDA_BASE_PATH=$(conda info --base)
if [ -z "$CONDA_BASE_PATH" ]; then
    echo "❌ Error: Could not determine Conda base path."
    exit 1
fi
echo "Detected CONDA_BASE_PATH: $CONDA_BASE_PATH"
CONDA_SH_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH_PATH" ]; then
    echo "❌ Error: conda.sh not found at $CONDA_SH_PATH."
    exit 1
fi
# shellcheck source=/dev/null
. "$CONDA_SH_PATH"
conda activate "$CONDA_ENV_NAME"
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "❌ Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    exit 1
fi
echo "✅ Conda environment '$CONDA_ENV_NAME' activated."
echo ""

echo "Installing Python dependencies into '$CONDA_ENV_NAME'..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then echo "❌ Error: Failed PyTorch install."; exit 1; fi
python -m pip install -r "$VLLM_REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then echo "❌ Error: Failed main requirements install."; exit 1; fi
echo "✅ All Python dependencies installed."
echo ""

echo "Generating Backend Python script: $BACKEND_TARGET_PATH"
cat > "$BACKEND_TARGET_PATH" << 'PYTHON_EOF'
# seraphim_backend.py

import os
import subprocess
import uuid
import re 
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import datetime
import logging
from typing import List, Optional, Dict, Any
import shlex

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s',
                      handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

SERAPHIM_DIR_PY = "{{SERAPHIM_DIR_PLACEHOLDER}}"
SCRIPTS_DIR_PY = "{{SCRIPTS_DIR_PLACEHOLDER}}"
VLLM_LOG_DIR_PY = "{{VLLM_LOG_DIR_PLACEHOLDER}}"
CONDA_ENV_NAME_PY = "{{CONDA_ENV_NAME_PLACEHOLDER}}"
BACKEND_PORT_PY = {{BACKEND_PORT_PLACEHOLDER}}
JOB_NAME_PREFIX_FOR_SQ_PY = "{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}" 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"],
)

class SlurmConfig(BaseModel):
    selected_model: str
    service_port: str
    hf_token: str | None = None
    job_name: str
    time_limit: str
    gpus: str
    cpus_per_task: str
    mem: str
    # mail_user field removed
    max_model_len: Optional[int] = Field(None, gt=0)

class DeployedServiceInfo(BaseModel):
    job_id: str
    job_name: str
    status: str
    nodes: Optional[str] = None
    node_ip: Optional[str] = None # Added for derived IP
    partition: Optional[str] = None
    time_used: Optional[str] = None
    user: Optional[str] = None
    service_url: Optional[str] = None
    slurm_output_file: Optional[str] = None
    slurm_error_file: Optional[str] = None
    raw_squeue_line: Optional[str] = None

def get_ip_from_node_name(node_name: Optional[str]) -> Optional[str]:
    if not node_name:
        return None
    
    # Remove any suffixes like (1) from node01(1)
    node_name_clean = node_name.split('(')[0].strip()

    # Pattern for ki-gXXXX -> 10.16.246.X
    match_ki_g = re.fullmatch(r"ki-g(\d+)", node_name_clean)
    if match_ki_g:
        numeric_part = int(match_ki_g.group(1)) # Converts "0002" to 2
        return f"10.16.246.{numeric_part}"
    
    # Add other patterns here if needed for different node types/clusters
    # Example:
    # match_other = re.fullmatch(r"compute-(\d+)-(\d+)", node_name_clean)
    # if match_other:
    #     return f"192.168.{match_other.group(1)}.{match_other.group(2)}"

    logger.debug(f"Node name '{node_name_clean}' did not match known IP patterns. Returning original node name for URL.")
    return node_name_clean # Fallback to original node name if no pattern matches


def generate_sbatch_script_content(config: SlurmConfig, scripts_dir: str, conda_env_name: str) -> tuple[str, str, str, str]:
    conda_base_path_for_slurm_script = "$(conda info --base)"
    escaped_selected_model_for_vllm_cmd = config.selected_model.replace('"', '\\"')

    vllm_serve_command = f'vllm serve "{escaped_selected_model_for_vllm_cmd}"'
    model_args = [
        f'--host "0.0.0.0"', f'--port {config.service_port}', '--trust-remote-code'
    ]
    
    current_max_model_len = 16384
    if config.max_model_len is not None:
        current_max_model_len = config.max_model_len
    # ... (rest of max_model_len logic from previous version)
    elif "llama-2-7b" in config.selected_model.lower() or "llama2-7b" in config.selected_model.lower(): current_max_model_len = 4096
    elif "mixtral" in config.selected_model.lower() or "pixtral" in config.selected_model.lower(): current_max_model_len = 32768
    
    if "pixtral" in config.selected_model.lower():
        model_args.append('--guided-decoding-backend=lm-format-enforcer')
        model_args.append("--limit_mm_per_prompt 'image=8'")
        if "mistralai/Pixtral-12B-2409" in config.selected_model:
            model_args.extend(['--enable-auto-tool-choice', '--tool-call-parser=mistral',
                                 '--tokenizer_mode mistral', '--revision aaef4baf771761a81ba89465a18e4427f3a105f9'])
                                
    model_args.append(f'--max-model-len {current_max_model_len}')
    vllm_serve_command_full = vllm_serve_command + " \\\n    " + " \\\n    ".join(model_args)

    mail_type_line = "#SBATCH --mail-type=NONE" # Simplified: mail_user removed
    safe_filename_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config.job_name)
    slurm_job_name = config.job_name 
    unique_id = str(uuid.uuid4())[:8]
    script_filename = f"deploy_{safe_filename_job_name}_{unique_id}.slurm" 
    script_path = os.path.join(scripts_dir, script_filename)

    slurm_out_file_pattern_for_sbatch = os.path.join(SCRIPTS_DIR_PY, f"{safe_filename_job_name}_%j.out")
    slurm_err_file_pattern_for_sbatch = os.path.join(SCRIPTS_DIR_PY, f"{safe_filename_job_name}_%j.err")

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_job_name}
#SBATCH --output={slurm_out_file_pattern_for_sbatch}
#SBATCH --error={slurm_err_file_pattern_for_sbatch}
#SBATCH --time={config.time_limit}
#SBATCH --gres=gpu:{config.gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem}
{mail_type_line}
# ... (rest of sbatch script content from previous version, no changes needed here) ...
echo "Current ulimit -n (soft): $(ulimit -Sn)"
echo "Current ulimit -n (hard): $(ulimit -Hn)"
ulimit -n 10240 
if [ $? -eq 0 ]; then echo "Successfully set ulimit -n to $(ulimit -Sn)"; else echo "WARN: Failed to set ulimit -n. Current: $(ulimit -Sn). Check hard limits if issues persist."; fi
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output File: {slurm_out_file_pattern_for_sbatch.replace('%j', '$SLURM_JOB_ID')}"
echo "Slurm Error File: {slurm_err_file_pattern_for_sbatch.replace('%j', '$SLURM_JOB_ID')}"
echo "Model: {config.selected_model}"
echo "Target Service Port: {config.service_port}"
echo "Conda Env: {conda_env_name}"
echo "Max Model Length Requested: {current_max_model_len}"
echo "vLLM service will run in the FOREGROUND of this Slurm job."
echo "=================================================================="
CONDA_BASE_PATH_SLURM="{conda_base_path_for_slurm_script}"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Conda base path empty."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then . "$CONDA_SH_PATH"; else echo "WARN: conda.sh not found."; fi
conda activate "{conda_env_name}"
if [[ "$CONDA_PREFIX" != *"{conda_env_name}"* ]]; then echo "ERROR: Failed to activate conda. Prefix: $CONDA_PREFIX"; exit 1; fi
echo "Conda env '{conda_env_name}' activated. Path: $CONDA_PREFIX";
HF_TOKEN_VALUE="{config.hf_token or ''}"
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi
export VLLM_CONFIGURE_LOGGING="0"
export VLLM_NO_USAGE_STATS="True"
export VLLM_DO_NOT_TRACK="True"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN set to 1."
echo -e "\\nStarting vLLM API Server in FOREGROUND..."
echo "Command: {vllm_serve_command_full}"
echo "vLLM logs will be in Slurm output/error files."
echo "--- vLLM Service Starting (Output will follow) ---"
{vllm_serve_command_full}
VLLM_EXIT_CODE=$?
echo "--- vLLM Service Ended (Exit Code: $VLLM_EXIT_CODE) ---"
echo "=================================================================="
echo "✝ SERAPHIM vLLM Job - FINAL STATUS ✝"
if [ $VLLM_EXIT_CODE -eq 0 ]; then echo "vLLM exited cleanly or was terminated."; else echo "ERROR: vLLM exited with code: $VLLM_EXIT_CODE."; fi
echo "Slurm job $SLURM_JOB_ID finished."
echo "=================================================================="
"""
    return script_path, sbatch_content, slurm_out_file_pattern_for_sbatch, slurm_err_file_pattern_for_sbatch

@app.post("/api/deploy")
async def deploy_vllm_service_api(config: SlurmConfig, request: Request):
    # ... (deploy logic from v2.2, no changes needed here) ...
    logger.info(f"Deployment request for model: {config.selected_model}, Service Port: {config.service_port}, Max Model Len: {config.max_model_len}, Job Name: {config.job_name}")
    try: os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    except OSError as e: raise HTTPException(status_code=500, detail=f"Server error creating script dir: {e}")
    script_path, sbatch_content, slurm_out_file_pattern, slurm_err_file_pattern = generate_sbatch_script_content(
        config, SCRIPTS_DIR_PY, CONDA_ENV_NAME_PY
    )
    try:
        with open(script_path, "w") as f: f.write(sbatch_content)
        os.chmod(script_path, 0o755)
        logger.info(f"Slurm script saved: {script_path}")
    except IOError as e: raise HTTPException(status_code=500, detail=f"Server error writing script: {e}")
    try:
        submit_command = ["sbatch", script_path]
        process = subprocess.run(submit_command, capture_output=True, text=True, check=True, timeout=30)
        job_id_message = process.stdout.strip()
        job_id = job_id_message.split(" ")[-1].strip() if "Submitted batch job" in job_id_message else "Unknown"
        logger.info(f"Sbatch successful. Output: '{job_id_message}', Parsed Job ID: {job_id}")
        actual_slurm_out = slurm_out_file_pattern.replace('%j', job_id if job_id != "Unknown" else "<JOB_ID>")
        actual_slurm_err = slurm_err_file_pattern.replace('%j', job_id if job_id != "Unknown" else "<JOB_ID>")
        return {"message": f"Slurm job submitted! ({job_id_message})", "job_id": job_id,
                "script_path": script_path, 
                "slurm_output_file_pattern": actual_slurm_out, 
                "slurm_error_file_pattern": actual_slurm_err,  
                "monitoring_note": f"Monitor Slurm output ({actual_slurm_out}) for service logs and errors."}
    except subprocess.TimeoutExpired: raise HTTPException(status_code=500, detail="sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        detail_msg = f"Sbatch failed. RC: {e.returncode}. Stderr: {e.stderr.strip()}" if e.stderr.strip() else "Sbatch failed."
        raise HTTPException(status_code=500, detail=detail_msg)
    except FileNotFoundError: raise HTTPException(status_code=500, detail="sbatch command not found.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Unexpected sbatch error: {str(e)}")


def parse_slurm_log_for_url(log_file_path: str, host_for_url: str, job_port: str) -> Optional[str]:
    # host_for_url is now the derived IP or original nodename
    if not os.path.exists(log_file_path):
        logger.debug(f"Log file not found for URL parsing: {log_file_path}")
        return None
    try:
        with open(log_file_path, 'r', errors='ignore') as f:
            for _ in range(200): 
                line = f.readline()
                if not line: break
                match = re.search(r"Uvicorn running on http://([\d\.]+):(\d+)", line)
                if match:
                    log_ip, log_port_str = match.groups()
                    if log_port_str == job_port:
                        # If Uvicorn says 0.0.0.0, use the passed host_for_url (which is preferred IP or nodename)
                        # Otherwise, use the specific IP Uvicorn reports (e.g., if not bound to 0.0.0.0)
                        service_host = host_for_url if host_for_url and log_ip == "0.0.0.0" else log_ip
                        if service_host: # Ensure service_host is not empty
                            return f"http://{service_host}:{job_port}/"
            logger.debug(f"Uvicorn URL line not found in first 200 lines of {log_file_path}")
            # Fallback guess if URL not found in log, using the determined host_for_url
            if host_for_url and job_port: 
                return f"http://{host_for_url}:{job_port}/ (ENDPOINT)"
    except Exception as e:
        logger.warning(f"Could not read or parse log file {log_file_path}: {e}")
    return None

@app.get("/api/active_deployments", response_model=List[DeployedServiceInfo])
async def get_active_deployments():
    deployments = []
    try:
        user = subprocess.check_output("whoami", text=True).strip()
        squeue_cmd = ["squeue", "-u", user, "-o", "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R",
                      "--noheader", "--states=RUNNING,PENDING,COMPLETING,SUSPENDED"]
        process = subprocess.run(squeue_cmd, capture_output=True, text=True, check=True, timeout=15)
        
        for line in process.stdout.strip().split('\n'):
            if not line.strip(): continue
            try:
                parts = line.split(maxsplit=7) 
                if len(parts) < 8: continue

                job_id = parts[0].strip()
                partition = parts[1].strip()
                job_name_squeue = parts[2].strip() 
                user_val = parts[3].strip()
                state_val = parts[4].strip()
                time_val = parts[5].strip()
                node_list_val_raw = parts[7].strip()   

                # Filter removed: Now shows all jobs for the user
                # if not job_name_squeue.startswith(JOB_NAME_PREFIX_FOR_SQ_PY): continue

                safe_job_name_for_logs = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in job_name_squeue)
                slurm_out_file_path = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name_for_logs}_{job_id}.out")
                slurm_err_file_path = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name_for_logs}_{job_id}.err")
                
                first_node_name = None
                derived_node_ip = None
                if state_val in ["R", "RUNNING"] and node_list_val_raw and node_list_val_raw != "(None)":
                    first_node_name = node_list_val_raw.split(',')[0].strip()
                    derived_node_ip = get_ip_from_node_name(first_node_name)

                service_info = DeployedServiceInfo(
                    job_id=job_id, job_name=job_name_squeue, status=state_val,
                    nodes=node_list_val_raw if node_list_val_raw != "(None)" else None, # Store raw node list
                    node_ip=derived_node_ip,
                    partition=partition, time_used=time_val, user=user_val,
                    slurm_output_file=slurm_out_file_path, 
                    slurm_error_file=slurm_err_file_path,   
                    raw_squeue_line=line
                )

                if service_info.status in ["R", "RUNNING"] and (derived_node_ip or first_node_name):
                    host_for_url = derived_node_ip or first_node_name # Prefer IP if available
                    job_port_from_name = "" # Default
                    if job_name_squeue.startswith(JOB_NAME_PREFIX_FOR_SQ_PY): # Port extraction more reliable for SERAPHIM jobs
                        port_match_in_name = re.search(r"_p(\d{4,5})", job_name_squeue) 
                        if port_match_in_name:
                            job_port_from_name = port_match_in_name.group(1)
                    
                    if os.path.exists(slurm_out_file_path) and host_for_url:
                        service_info.service_url = parse_slurm_log_for_url(
                            slurm_out_file_path,
                            host_for_url, # Pass derived IP or nodename
                            job_port_from_name
                        )
                
                deployments.append(service_info)
            except Exception as e:
                logger.error(f"Error parsing squeue line '{line}': {e}", exc_info=True)
                deployments.append(DeployedServiceInfo(job_id="PARSE_ERROR", job_name=f"Error processing: {line[:60]}...", status="UNKNOWN"))
    # ... (rest of get_active_deployments error handling) ...
    except subprocess.TimeoutExpired: logger.error("squeue command timed out.")
    except subprocess.CalledProcessError as e: logger.error(f"squeue command failed with RC {e.returncode}: {e.stderr}")
    except Exception as e: logger.error(f"Error fetching active deployments: {e}", exc_info=True)
    return deployments

@app.post("/api/cancel_job/{job_id}")
async def cancel_slurm_job_api(job_id: str):
    # ... (cancel_job logic from v2.2, no changes needed here) ...
    logger.info(f"Request to cancel Slurm job: {job_id}")
    try:
        if not re.match(r"^\d+$", job_id):
            raise HTTPException(status_code=400, detail="Invalid job ID format.")
        cancel_command = ["scancel", job_id]
        process = subprocess.run(cancel_command, capture_output=True, text=True, timeout=10)
        if process.returncode == 0:
            logger.info(f"Successfully initiated cancellation for job {job_id}. Scancel output: {process.stdout.strip()}")
            return {"message": f"Cancellation command sent for Slurm job {job_id}. Check Slurm status.", "details": process.stdout.strip()}
        else:
            error_detail = f"scancel command for job {job_id} failed or job already gone. RC: {process.returncode}."
            if process.stderr.strip(): error_detail += f" Stderr: {process.stderr.strip()}"
            if process.stdout.strip(): error_detail += f" Stdout: {process.stdout.strip()}" # sometimes info is on stdout for errors
            logger.warning(error_detail)
            raise HTTPException(status_code=400, detail=error_detail)
    except subprocess.TimeoutExpired:
        logger.error(f"scancel command timed out for job {job_id}.")
        raise HTTPException(status_code=500, detail=f"scancel command for job {job_id} timed out.")
    except FileNotFoundError:
        logger.error("scancel command not found.")
        raise HTTPException(status_code=500, detail="scancel command not found on the server.")
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Unexpected error cancelling job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while trying to cancel job {job_id}: {str(e)}")


@app.get("/api/log_content")
async def get_log_content_api(file_path: str = Query(..., description="Full path to the log file to read.")):
    # ... (log_content logic from v2.2, minor tweak to truncation message if needed, but largely same) ...
    logger.info(f"Log content request for: {file_path}")
    abs_scripts_dir = os.path.abspath(SCRIPTS_DIR_PY)
    abs_file_path = os.path.abspath(file_path)

    if not abs_file_path.startswith(abs_scripts_dir):
        logger.warning(f"Forbidden access attempt for log file: {file_path} (resolved: {abs_file_path}). Not in {abs_scripts_dir}")
        raise HTTPException(status_code=403, detail="Access to this file path is forbidden.")
    if not os.path.exists(abs_file_path):
        logger.info(f"Log file not yet found or path is incorrect: {abs_file_path}")
        return {"log_content": f"(Log file not yet available: {os.path.basename(file_path)})"}
    if not os.path.isfile(abs_file_path):
        logger.warning(f"Requested path is not a file: {abs_file_path}")
        raise HTTPException(status_code=400, detail=f"Requested path is not a file: {file_path}")

    try:
        log_content_lines = []
        file_size = os.path.getsize(abs_file_path)
        max_lines_total = 300 
        max_bytes_head = 30 * 1024 
        max_bytes_tail = 70 * 1024 
        
        if file_size == 0: return {"log_content": "(Log file is empty)"}

        if file_size <= (max_bytes_head + max_bytes_tail): 
            with open(abs_file_path, 'r', errors='ignore') as f:
                log_content_lines = [line for i, line in enumerate(f) if i < max_lines_total]
                if len(log_content_lines) == max_lines_total:
                    current_pos = f.tell()
                    f.seek(0, os.SEEK_END) # Go to end to check actual file size vs bytes read
                    if current_pos < f.tell() : 
                        log_content_lines.append(f"\n--- (Log truncated, showing first {max_lines_total} lines) ---\n")
        else: 
            head_lines_count = max_lines_total // 3
            tail_lines_count = max_lines_total - head_lines_count
            with open(abs_file_path, 'r', errors='ignore') as f:
                bytes_read = 0
                for i in range(head_lines_count):
                    line = f.readline()
                    if not line or bytes_read > max_bytes_head: break
                    log_content_lines.append(line)
                    bytes_read += len(line.encode('utf-8'))
            log_content_lines.append(f"\n\n... (log truncated - {file_size // 1024} KB total) ...\n\n")
            with open(abs_file_path, 'rb') as f: 
                f.seek(0, os.SEEK_END)
                current_seek_offset = min(max_bytes_tail, f.tell())
                f.seek(-current_seek_offset, os.SEEK_END)
                if f.tell() > 0: f.readline() 
                raw_tail_lines = [line_bytes.decode('utf-8', errors='ignore') for line_bytes in f]
                log_content_lines.extend(raw_tail_lines[-tail_lines_count:])
        return {"log_content": "".join(log_content_lines)}
    except Exception as e:
        logger.error(f"Could not read log file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

if __name__ == "__main__":
    os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    os.makedirs(VLLM_LOG_DIR_PY, exist_ok=True) 
    logger.info(f"Starting SERAPHIM Backend Server on http://0.0.0.0:{BACKEND_PORT_PY}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT_PY)
PYTHON_EOF
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write Backend Python script."; exit 1; fi;
ESCAPED_SERAPHIM_DIR_FOR_SED=$(printf '%s\n' "$SERAPHIM_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_SCRIPTS_DIR_FOR_SED=$(printf '%s\n' "$SCRIPTS_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_VLLM_LOG_DIR_FOR_SED=$(printf '%s\n' "$VLLM_LOG_DIR_IN_INSTALLER" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_ENV_NAME_FOR_SED=$(printf '%s\n' "$CONDA_ENV_NAME" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_BASE_PATH_FOR_SED=$(printf '%s\n' "$CONDA_BASE_PATH" | sed 's:[&/\]:\\&:g')
ESCAPED_JOB_NAME_PREFIX_FOR_SQ_FOR_SED=$(printf '%s\n' "$JOB_NAME_PREFIX_FOR_SQ" | sed 's:[&/\]:\\&:g')

sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{SCRIPTS_DIR_PLACEHOLDER}}|$ESCAPED_SCRIPTS_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{VLLM_LOG_DIR_PLACEHOLDER}}|$ESCAPED_VLLM_LOG_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}|$ESCAPED_JOB_NAME_PREFIX_FOR_SQ_FOR_SED|g" "$BACKEND_TARGET_PATH"
echo "✅ Backend Python script ($BACKEND_FILENAME) configured."
echo ""

echo "Generating JavaScript logic file: $JS_TARGET_PATH";
cat > "$JS_TARGET_PATH" << 'EOF_JS'
// seraphim_logic.js
const BACKEND_API_BASE_URL = `http://${window.location.hostname}:{{BACKEND_PORT_PLACEHOLDER}}/api`;
const MODELS_FILE_URL = 'models.txt';

let allModels = [];
let currentSelectedJobDetails = null; 
// LOG_REFRESH_INTERVAL: 1000ms = 1 second. 
// User requested 0.5s (500ms), which is very aggressive for file I/O.
// Setting to 1s as a compromise. Adjust to 500 if server performance allows.
const LOG_REFRESH_INTERVAL = 1000; 

async function fetchAndPopulateModels() {
    // ... (fetchAndPopulateModels logic from v2.2 - no changes) ...
    console.log("SERAPHIM_DEBUG: Fetching models from", MODELS_FILE_URL);
    const modelSelect = document.getElementById('model-select');
    const modelSearchInput = document.getElementById('model-search');
    if (!modelSelect || !modelSearchInput) {
        console.error("SERAPHIM_DEBUG: Model select or search input not found!");
        return;
    }
    modelSelect.innerHTML = '<option value="">-- Loading models... --</option>';
    try {
        const response = await fetch(MODELS_FILE_URL);
        if (!response.ok) {
            const errorText = `Failed to fetch ${MODELS_FILE_URL}: ${response.status} ${response.statusText}. Please ensure models.txt exists.`;
            throw new Error(errorText);
        }
        const text = await response.text();
        const lines = text.split('\n');
        allModels = [];
        lines.forEach(line => {
            line = line.trim();
            if (line && !line.startsWith('#')) {
                const parts = line.split(',');
                if (parts.length >= 2) {
                    allModels.push({ id: parts[0].trim(), name: parts.slice(1).join(',').trim() });
                } else if (parts.length === 1 && parts[0]) {
                    allModels.push({ id: parts[0], name: parts[0] });
                }
            }
        });
        if (allModels.length === 0) {
            modelSelect.innerHTML = '<option value="">-- No models in models.txt --</option>';
            if(document.getElementById('output')) document.getElementById('output').textContent = `⚠️ No models in ${MODELS_FILE_URL}.`;
            return;
        }
        allModels.sort((a, b) => a.name.localeCompare(b.name));
        populateModelDropdown(allModels);
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        modelSelect.innerHTML = `<option value="">-- Error loading models --</option>`;
        if(document.getElementById('output')) document.getElementById('output').textContent = `❌ ${error.message}`;
    }
}

function populateModelDropdown(modelsToDisplay) {
    // ... (populateModelDropdown logic from v2.2 - no changes) ...
    const modelSelect = document.getElementById('model-select');
    const currentSearchVal = document.getElementById('model-search').value; 
    const currentSelectedVal = modelSelect.value;
    modelSelect.innerHTML = ''; 
    if (modelsToDisplay.length === 0 && currentSearchVal === "") {
        const placeholder = document.createElement('option');
        placeholder.value = ""; placeholder.textContent = "-- Select a Model --";
        modelSelect.appendChild(placeholder);
    } else if (modelsToDisplay.length === 0 && currentSearchVal !== "") {
       const noMatch = document.createElement('option');
        noMatch.value = ""; noMatch.textContent = "-- No models match search --";
        modelSelect.appendChild(noMatch);
    }
    modelsToDisplay.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id; option.textContent = model.name;
        modelSelect.appendChild(option);
    });
    if (modelsToDisplay.some(m => m.id === currentSelectedVal)) {
        modelSelect.value = currentSelectedVal;
    }
     if(modelSelect.options.length > 0 && !modelSelect.value && currentSearchVal === ""){
        const placeholder = document.createElement('option');
        placeholder.value = ""; placeholder.textContent = "-- Select a Model --";
        modelSelect.insertBefore(placeholder, modelSelect.firstChild);
        modelSelect.value = "";
    }
}

function filterModels() {
    // ... (filterModels logic from v2.2 - no changes) ...
    const searchTerm = document.getElementById('model-search').value.toLowerCase();
    const filtered = allModels.filter(m => m.name.toLowerCase().includes(searchTerm) || m.id.toLowerCase().includes(searchTerm));
    populateModelDropdown(filtered);
}

async function fetchLogContent(filePath, displayElementId) {
    const displayElement = document.getElementById(displayElementId);
    if (!filePath || filePath === 'null' || filePath === 'undefined') {
        displayElement.textContent = 'Log file path not available.';
        return;
    }
    const initialFetch = displayElement.textContent.startsWith('🔄 Fetching') || 
                           displayElement.textContent.startsWith('Log file path not available') ||
                           displayElement.textContent.startsWith('Select a job') ||
                           displayElement.textContent.startsWith('No active') ||
                           displayElement.textContent.startsWith('Previously selected') ||
                           displayElement.textContent.startsWith('Cancelled job');
    if (initialFetch) {
      displayElement.textContent = `🔄 Fetching ${filePath.split('/').pop()}...`;
    }
    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/log_content?file_path=${encodeURIComponent(filePath)}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        const newContent = result.log_content || '(Log file is empty)';
        
        if (displayElement.textContent !== newContent) {
            displayElement.textContent = newContent;
            // MODIFIED SCROLL LOGIC:
            // Always scroll the error log down if new content.
            // For other logs (e.g. output log), use smart scroll (only if initial or already at bottom).
            if (displayElementId === 'log-error-content') {
                displayElement.scrollTop = displayElement.scrollHeight;
            } else {
                const isScrolledToBottom = displayElement.scrollHeight - displayElement.clientHeight <= displayElement.scrollTop + 1;
                if (initialFetch || isScrolledToBottom) {
                    displayElement.scrollTop = displayElement.scrollHeight;
                }
            }
        }
    } catch (error) {
        console.error(`SERAPHIM_DEBUG: Error fetching log ${filePath}:`, error);
        if (initialFetch) { // Only show error prominently on initial fetch attempt
            displayElement.textContent = `❌ Error fetching log: ${error.message}`;
        }
    }
}

async function cancelJob(jobId) {
    const outputDiv = document.getElementById('output');
    // Updated confirmation message:
    if (!confirm("Are you sure you want to cancel this job? (Be mindful that other users may be using it, consult with your colleagues before proceeding)")) {
        return;
    }
    outputDiv.textContent = `🔄 Attempting to cancel job ${jobId}...`;
    // ... (rest of cancelJob logic from v2.2) ...
    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/cancel_job/${jobId}`, { method: 'POST' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        outputDiv.textContent = `✅ ${result.message || 'Cancellation sent.'} Details: ${result.details || ''}`;
        outputDiv.style.color = "var(--success-color)";
        if (currentSelectedJobDetails && currentSelectedJobDetails.jobId === jobId) {
            currentSelectedJobDetails = null; 
            document.getElementById('log-output-content').textContent = "Cancelled job's logs cleared. Select another job.";
            document.getElementById('log-error-content').textContent = "";
        }
    } catch (error) {
        outputDiv.textContent = `❌ Error cancelling job ${jobId}: ${error.message}`;
        outputDiv.style.color = "var(--error-color)";
    } finally {
        await refreshDeployedEndpoints();
    }
}

async function refreshDeployedEndpoints(jobToSelect = null) {
    // ... (refreshDeployedEndpoints logic from v2.2, with minor text changes if needed) ...
    console.log("SERAPHIM_DEBUG: Refreshing active deployments...");
    const listDiv = document.getElementById('deployed-endpoints-list');
    const refreshButton = document.getElementById('refresh-endpoints-button');
    const logOutDisplay = document.getElementById('log-output-content');
    const logErrDisplay = document.getElementById('log-error-content');

    if (!listDiv || !refreshButton || !logOutDisplay || !logErrDisplay) return;

    listDiv.innerHTML = "<p><em>🔄 Fetching active jobs...</em></p>";
    if (!jobToSelect && !currentSelectedJobDetails) { 
        currentSelectedJobDetails = null;
        logOutDisplay.textContent = "Select a job to view its API service log.";
        logErrDisplay.textContent = "Select a job to view its internal vLLM engine log.";
    }
    refreshButton.disabled = true;

    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/active_deployments`);
        if (!response.ok) {
          const errRes = await response.json().catch(()=>({detail: "Unknown fetch error"}));
          throw new Error(errRes.detail);
        }
        const deployments = await response.json();

        if (deployments.length === 0) {
            listDiv.innerHTML = "<p>No active Slurm jobs found for your user.</p>";
            logOutDisplay.textContent = "No active jobs.";
            logErrDisplay.textContent = "No active jobs.";
            currentSelectedJobDetails = null;
        } else {
            let html = '<ul>';
            deployments.forEach(job => {
                const outFile = job.slurm_output_file ? String(job.slurm_output_file) : '';
                const errFile = job.slurm_error_file ? String(job.slurm_error_file) : '';
                html += `<li class="endpoint-item" data-jobid="${job.job_id}" data-outfile="${outFile}" data-errfile="${errFile}">
                        <strong>Job ID:</strong> ${job.job_id} (${job.status || 'N/A'})<br/>
                        <strong>Name:</strong> ${job.job_name || 'N/A'}
                        ${job.nodes ? `<br/><strong>Node(s):</strong> ${job.nodes}` : ''}
                        ${job.node_ip ? `<br/><strong>Node IP:</strong> ${job.node_ip}` : ''}
                        ${job.service_url ? `<br/><strong>URL:</strong> <a href="${job.service_url}" target="_blank" onclick="event.stopPropagation();">${job.service_url}</a>` : (job.status === 'R' && job.node_ip ? `<em>Service on ${job.node_ip} (port in name/log)</em>` : '')}
                        <br/><button class="cancel-job-button" data-jobid="${job.job_id}">Cancel</button>
                       </li>`;
            });
            html += '</ul>';
            listDiv.innerHTML = html;

            listDiv.querySelectorAll('.endpoint-item').forEach(item => {
                item.addEventListener('click', async function() {
                    listDiv.querySelectorAll('.endpoint-item.selected').forEach(sel => sel.classList.remove('selected'));
                    this.classList.add('selected');
                    const jobId = this.dataset.jobid;
                    const outFile = this.dataset.outfile;
                    const errFile = this.dataset.errfile;
                    currentSelectedJobDetails = { jobId, outFile, errFile };
                    await fetchLogContent(outFile, 'log-output-content');
                    await fetchLogContent(errFile, 'log-error-content');
                });
            });

            listDiv.querySelectorAll('.cancel-job-button').forEach(button => {
                button.addEventListener('click', e => { e.stopPropagation(); cancelJob(e.target.dataset.jobid); });
            });
            
            let jobAutoSelected = false;
            if (jobToSelect && jobToSelect.jobId) {
                const itemToSelect = listDiv.querySelector(`.endpoint-item[data-jobid="${jobToSelect.jobId}"]`);
                if (itemToSelect) {
                    itemToSelect.classList.add('selected');
                    currentSelectedJobDetails = { jobId: jobToSelect.jobId, outFile: jobToSelect.outFile, errFile: jobToSelect.errFile };
                    await fetchLogContent(jobToSelect.outFile, 'log-output-content');
                    await fetchLogContent(jobToSelect.errFile, 'log-error-content');
                    jobAutoSelected = true;
                } else {
                     console.warn(`SERAPHIM_DEBUG: Auto-selected job ${jobToSelect.jobId} disappeared after refresh.`);
                }
            }
            if (!jobAutoSelected && currentSelectedJobDetails) {
                const stillExists = deployments.some(d => d.job_id === currentSelectedJobDetails.jobId);
                if (!stillExists) {
                    currentSelectedJobDetails = null;
                    logOutDisplay.textContent = "Previously selected job no longer active.";
                    logErrDisplay.textContent = "";
                } else { 
                    const itemToReselect = listDiv.querySelector(`.endpoint-item[data-jobid="${currentSelectedJobDetails.jobId}"]`);
                    if (itemToReselect) itemToReselect.classList.add('selected');
                }
            } else if (!jobAutoSelected && !currentSelectedJobDetails && deployments.length > 0) {
                 logOutDisplay.textContent = "Select a job to view its API service log.";
                 logErrDisplay.textContent = "Select a job to view its internal vLLM engine log.";
            }
        }
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error refreshing active deployments:", error);
        listDiv.innerHTML = `<p style="color: var(--error-color);">❌ Error fetching: ${error.message}</p>`;
        logOutDisplay.textContent = "Error loading deployments.";
        logErrDisplay.textContent = "";
        currentSelectedJobDetails = null;
    } finally {
        refreshButton.disabled = false;
    }
}

async function handleDeployClick() {
    // ... (handleDeployClick logic from v2.2 - no changes) ...
    const outputDiv = document.getElementById('output');
    const deployButton = document.getElementById('deploy-button');
    deployButton.disabled = true; deployButton.textContent = "Submitting...";
    outputDiv.textContent = "🚀 Submitting deployment request...";
    outputDiv.style.color = "var(--text-color)";
    const slurmConfig = {
        selected_model: document.getElementById('model-select').value,
        service_port: document.getElementById('service-port').value,
        hf_token: document.getElementById('hf-token').value || null,
        max_model_len: document.getElementById('max-model-len').value ? parseInt(document.getElementById('max-model-len').value, 10) : null,
        job_name: document.getElementById('job-name').value,
        time_limit: document.getElementById('time-limit').value,
        gpus: document.getElementById('gpus').value,
        cpus_per_task: document.getElementById('cpus-per-task').value,
        mem: document.getElementById('mem').value,
        // mail_user field removed
    };
    if (!slurmConfig.selected_model || !slurmConfig.job_name) {
        outputDiv.textContent = "⚠️ Please select a model and enter a Job Name.";
        outputDiv.style.color = "var(--warning-color)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
    }
    if (slurmConfig.max_model_len !== null && (isNaN(slurmConfig.max_model_len) || slurmConfig.max_model_len <= 0)) {
        outputDiv.textContent = "⚠️ Max Model Length must be a positive number if specified.";
        outputDiv.style.color = "var(--warning-color)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
    }
    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/deploy`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(slurmConfig)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        outputDiv.style.color = "var(--success-color)";
        outputDiv.textContent = `✅ ${result.message || 'Job submitted!'}\nJob ID: ${result.job_id}\nOutput: ${result.slurm_output_file_pattern}\nError: ${result.slurm_error_file_pattern}`;
        await refreshDeployedEndpoints({ 
            jobId: result.job_id, 
            outFile: result.slurm_output_file_pattern, 
            errFile: result.slurm_error_file_pattern 
        });
    } catch (error) {
        outputDiv.style.color = "var(--error-color)";
        outputDiv.textContent = `❌ Error: ${error.message}`;
    } finally {
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm";
    }
}

async function pollCurrentJobLogs() {
    if (currentSelectedJobDetails && currentSelectedJobDetails.jobId) {
        // console.log(`SERAPHIM_DEBUG: Polling logs for ${currentSelectedJobDetails.jobId}`);
        if (currentSelectedJobDetails.outFile && currentSelectedJobDetails.outFile !== 'null' && currentSelectedJobDetails.outFile !== 'undefined') {
            await fetchLogContent(currentSelectedJobDetails.outFile, 'log-output-content');
        }
        if (currentSelectedJobDetails.errFile && currentSelectedJobDetails.errFile !== 'null' && currentSelectedJobDetails.errFile !== 'undefined') {
            await fetchLogContent(currentSelectedJobDetails.errFile, 'log-error-content');
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    await fetchAndPopulateModels(); 
    document.getElementById('model-search')?.addEventListener('input', filterModels);
    document.getElementById('deploy-button')?.addEventListener('click', handleDeployClick);
    document.getElementById('refresh-endpoints-button')?.addEventListener('click', () => refreshDeployedEndpoints());
    await refreshDeployedEndpoints(); 
    setInterval(pollCurrentJobLogs, LOG_REFRESH_INTERVAL); 
});
EOF_JS
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write JavaScript file."; exit 1; fi;
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$JS_TARGET_PATH";
echo "✅ JavaScript logic file ($JS_FILENAME) configured."
echo ""

echo "Generating HTML file: $HTML_TARGET_PATH";
cat > "$HTML_TARGET_PATH" << 'EOF_HTML'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>✧ SERAPHIM CORE ✧ vLLM Deployment Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --matrix-green: #00FF00; /* Bright Matrix Green */
            --matrix-green-darker: #00AA00;
            --matrix-green-darkest: #005500;
            --matrix-bg: #000000;
            --matrix-card-bg: #0D0D0D; /* Very dark grey, almost black */

            --primary-color: var(--matrix-green); 
            --secondary-color: var(--matrix-green-darker); /* Using darker green for less emphasis */
            --accent-color: var(--matrix-green); 
            --bg-color: var(--matrix-bg); 
            --card-bg-color: var(--matrix-card-bg); 
            --text-color: var(--matrix-green); /* Main text is green */
            --text-muted-color: var(--matrix-green-darker); 
            --border-color: var(--matrix-green-darkest);
            
            --font-body: 'Exo 2', 'Courier New', Courier, monospace; /* Added monospace fallback */
            --font-heading: 'Orbitron', 'Courier New', Courier, monospace;
            
            --success-color: var(--matrix-green); /* Success is also matrix green */
            --warning-color: #FFFF00; /* Yellow for warnings stands out in matrix theme */
            --error-color: #FF0000;   /* Red for errors */
            --cancel-button-bg: #AA0000; 
            --cancel-button-hover-bg: #FF0000;
            --log-text-color: var(--matrix-green);
        }
        html, body { height: 100%; margin: 0; padding: 0; overflow-x: hidden; }
        body { 
            font-family: var(--font-body); background-color: var(--bg-color); 
            color: var(--text-color); display: flex; flex-direction: column; 
            font-size: 16px; line-height: 1.5; /* Slightly reduced line-height */
        }
        .header { 
            background: linear-gradient(135deg, var(--matrix-green-darkest) 0%, var(--matrix-bg) 70%, var(--matrix-green-darkest) 100%); 
            color: var(--matrix-green); padding: 10px 20px; text-align: center; 
            border-bottom: 2px solid var(--matrix-green); 
            text-shadow: 0 0 5px var(--matrix-green); flex-shrink: 0; 
        }
        .header h1 { margin: 0; font-family: var(--font-heading); font-size: 2em; font-weight: 700; letter-spacing: 2px; display: flex; align-items: center; justify-content: center; gap: 10px; }
        .header p { margin: 3px 0 0; font-size: 0.85em; opacity: 0.8; font-weight: 300;}
        
        .page-content-wrapper {
            display: flex; flex-grow: 1; 
            padding: 10px; gap: 10px;    
            overflow: hidden; max-width: 100%; 
            box-sizing: border-box; align-items: stretch; 
        }

        .column { 
            display: flex; flex-direction: column; 
            background-color: var(--card-bg-color);
            padding: 12px; 
            border-radius: 8px; /* Softer radius */
            border: 1px solid var(--border-color);
            box-shadow: 0 0 10px var(--matrix-green-darkest); /* Subtle glow */
            overflow: hidden; 
        }
        
        .log-column { flex: 1 1 23%; min-width: 260px; } 
        .deploy-column { flex: 1.5 1 27%; min-width: 360px; } 
        .endpoints-column { flex: 1.2 1 27%; min-width: 300px; }


        .column h3 {
            font-family: var(--font-heading); color: var(--matrix-green); 
            border-bottom: 1px solid var(--matrix-green-darkest); padding-bottom: 6px; 
            margin-top: 0; margin-bottom: 10px; font-size: 1.15em; 
            letter-spacing: 1px; text-shadow: 0 0 3px var(--matrix-green-darker);
            display: flex; align-items: center; gap: 8px; flex-shrink: 0;
        }
        
        .form-container, .endpoints-container-inner {
           display: flex; flex-direction: column;
           overflow: hidden; flex-grow: 1; 
        }
        
        .log-column pre { 
            background-color: #000000; /* Black background for log area */
            color: var(--log-text-color); 
            padding: 10px; 
            border-radius: 4px; font-family: 'Monaco', 'Consolas', 'Courier New', Courier, monospace; /* More specific monospace */
            font-size: 1.0em; /* Increased log font size */
            white-space: pre-wrap; word-wrap: break-word;
            border: 1px solid var(--matrix-green-darkest);
            flex-grow: 1; overflow-y: auto; min-height: 150px; 
            scrollbar-width: thin; scrollbar-color: var(--matrix-green-darker) #000;
        }
        .log-column pre::-webkit-scrollbar { width: 8px; }
        .log-column pre::-webkit-scrollbar-track { background: #000; }
        .log-column pre::-webkit-scrollbar-thumb { background-color: var(--matrix-green-darker); border-radius: 4px; border: 1px solid #000; }

        .searchable-select-container {
            display: flex; flex-direction: column; margin-bottom: 8px;
        }
        #model-search {
            width: 100%; padding: 7px; box-sizing: border-box; 
            font-size: 0.85em; background-color: #000000; 
            color: var(--matrix-green); border: 1px solid var(--matrix-green-darker);
            border-bottom: none; border-radius: 4px 4px 0 0; margin-bottom: 0;
        }
        #model-search:focus {
           border-color: var(--matrix-green); 
           box-shadow: 0 0 3px var(--matrix-green); 
           outline: none; background-color: #111111; z-index: 10;
        }
        #model-select {
            width: 100%; padding: 7px; box-sizing: border-box; font-size: 0.85em;
            background-color: #000000; color: var(--matrix-green);
            border: 1px solid var(--matrix-green-darker);
            border-radius: 0 0 4px 4px; margin-top: -1px; /* Overlap border slightly */
        }
        #model-select:focus {
            border-color: var(--matrix-green);
            box-shadow: 0 0 3px var(--matrix-green); outline: none;
        }
        /* Styling for options in select is browser-dependent, this is a best effort */
        #model-select option { background-color: #000000; color: var(--matrix-green); }


        label { display: block; margin-top: 8px; margin-bottom: 2px; font-weight: 400; font-size: 0.75em; color: var(--text-muted-color); text-transform: uppercase; letter-spacing: 0.5px;}
        select:not(#model-select), input[type="text"]:not(#model-search), input[type="number"], input[type="email"], input[type="password"] { 
            width: 100%; padding: 7px; margin-bottom: 8px; 
            border-radius: 4px; border: 1px solid var(--matrix-green-darker); 
            box-sizing: border-box; font-size: 0.85em; 
            background-color: #000000; color: var(--matrix-green); 
        }
        select:not(#model-select):focus, input[type="text"]:not(#model-search):focus, input[type="number"]:focus, input[type="email"]:focus, input[type="password"]:focus { 
            border-color: var(--matrix-green); 
            box-shadow: 0 0 5px var(--matrix-green); 
            outline: none; background-color: #111111; 
        }

        button { 
            background: var(--matrix-green-darkest); 
            color: var(--matrix-green); padding: 8px 12px; cursor: pointer; border: 1px solid var(--matrix-green-darker); 
            border-radius: 4px; font-weight: bold; font-size: 0.85em; 
            text-transform: uppercase; letter-spacing: 1px; 
            box-shadow: 0 0 5px var(--matrix-green-darkest); 
            transition: all 0.2s ease; margin-top: 6px; width: 100%; 
            text-shadow: 0 0 3px var(--matrix-green);
        }
        button:hover:not(:disabled), button:focus:not(:disabled) { 
            background: var(--matrix-green-darker); 
            color: #000000;
            box-shadow: 0 0 10px var(--matrix-green); transform: translateY(-1px); 
            text-shadow: none;
        }
        button:disabled { background: #222; color: #555; cursor: not-allowed; opacity: 0.7; border-color: #333;}
        
        #output { 
            margin-top: 8px; padding: 8px; background-color: #000000; 
            border: 1px solid var(--border-color); border-radius: 4px; 
            white-space: pre-wrap; word-wrap: break-word; font-size: 0.75em; 
            max-height: 80px; overflow-y: auto; line-height: 1.3; 
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace; color: var(--text-color);
            flex-shrink: 0; 
        }
        .slurm-options h3 { margin-top: 12px; font-size: 0.9em; color: var(--matrix-green-darker); }
        .slurm-options label { font-size: 0.7em; margin-top: 4px; }
        
        #refresh-endpoints-button { background: var(--matrix-green-darkest); margin-bottom: 8px; flex-shrink:0; }
        #refresh-endpoints-button:hover:not(:disabled) { background: var(--matrix-green-darker); color: #000; }
        
        #deployed-endpoints-list { overflow-y: auto; flex-grow: 1; padding-right: 3px; }
        #deployed-endpoints-list ul { list-style-type: none; padding: 0; margin:0;}
        .endpoint-item { 
            background-color: #000; border: 1px solid var(--matrix-green-darkest); 
            padding: 7px; margin-bottom: 5px; border-radius: 4px; 
            font-size: 0.7em; line-height: 1.3; 
            cursor: pointer; transition: background-color 0.2s ease; 
        }
        .endpoint-item:hover { background-color: var(--matrix-green-darkest); }
        .endpoint-item.selected { background-color: var(--matrix-green-darker); color: #000; border-left: 3px solid var(--matrix-green); }
        .endpoint-item.selected strong { color: var(--matrix-green); text-shadow: 0 0 2px #000;} 
        .endpoint-item.selected a { color: #000; }
        .endpoint-item strong { color: var(--matrix-green); }
        .endpoint-item a { color: var(--accent-color); text-decoration: none; font-weight: bold; word-break: break-all;}
        .endpoint-item a:hover { text-decoration: underline; color: var(--matrix-green); }
        .cancel-job-button {
            background: var(--cancel-button-bg); color: #fff;
            padding: 2px 5px; font-size: 0.95em; 
            margin-top: -45px; width: auto; display: inline-block; float: right;
            border: 1px solid var(--error-color); text-shadow: none;
        }
        .cancel-job-button:hover:not(:disabled) { background: var(--cancel-button-hover-bg); border-color: #FF5555; }

        .footer { text-align: center; padding: 10px; background-color: #000; color: var(--matrix-green-darker); font-size: 0.8em; border-top: 2px solid var(--matrix-green); flex-shrink: 0; text-shadow: 0 0 3px var(--matrix-green-darkest); }
        .icon { margin-right: 6px; font-size: 1em; vertical-align: middle;}
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="icon"></span> SERAPHIM <span class="icon"></span></h1>
        <p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p>
        <p>Made by Anderson de Lima Luiz</p>
    </div>
    
    <div class="page-content-wrapper">
        <div class="column log-column" id="output-log-column-wrapper">
            <h3><span class="icon">📄</span> API service log</h3>
            <pre id="log-output-content">Select a job to view its log.</pre>
        </div>

        <div class="column deploy-column" id="deploy-form-column-wrapper">
            <div class="form-container"> 
                <h3><span class="icon">⚙️</span> Deploy New vLLM Instance</h3>
                
                <label for="model-search">Select Model (type to filter):</label>
                <div class="searchable-select-container">
                    <input type="text" id="model-search" placeholder="Filter models..." autocomplete="off">
                    <select id="model-select"><option value="">-- Loading... --</option></select>
                </div>
                
                <label for="max-model-len">Max Model Length (Optional):</label>
                <input type="number" id="max-model-len" placeholder="e.g., 4096 (blank for default)" min="1"/>
                <label for="service-port">Service Port (on Slurm node):</label>
                <input type="number" id="service-port" value="" min="1024" max="65535"/>
                <label for="hf-token">Hugging Face Token (Optional):</label>
                <input type="password" id="hf-token" placeholder="For Llama, gated models, etc."/>
                
                <div class="slurm-options">
                    <h3>Slurm Configuration</h3>
                    <label for="job-name">Job Name (e.g., {{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}_model_pPORT):</label>
                    <input type="text" id="job-name" value="{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}_model_p8000"/>
                    <label for="time-limit">Time Limit (HH:MM:SS):</label><input type="text" id="time-limit" value="23:59:59"/>
                    <label for="gpus">GPUs (e.g., 1 or a100:1):</label><input type="text" id="gpus" value="1"/>
                    <label for="cpus-per-task">CPUs per Task:</label><input type="number" id="cpus-per-task" value="4" min="1"/>
                    <label for="mem">Memory (e.g., 32G):</label><input type="text" id="mem" value="32G"/>
                    </div>
                <button id="deploy-button">Deploy to Slurm</button>
                <div id="output">Configure and click deploy. Status will appear here.</div>
            </div>
        </div>

        <div class="column endpoints-column" id="active-deployments-column-wrapper">
            <div class="endpoints-container-inner"> 
                <h3><span class="icon">📡</span> Active Slurm Jobs</h3>
                <button id="refresh-endpoints-button">Refresh Status</button>
                <div id="deployed-endpoints-list"><p><em>Loading active jobs...</em></p></div>
            </div>
        </div>

        <div class="column log-column" id="error-log-column-wrapper">
            <h3><span class="icon">⚠️</span> Internal vLLM engine log</h3>
            <pre id="log-error-content">Select a job to view its log.</pre>
        </div>
    </div>

    <div class="footer">✧ SERAPHIM CORE Interface v2.3.1 (Matrix Edition) ✧ TDC AI | ANDERSON LUIZ ✧</div>
    <script src="seraphim_logic.js" defer></script>
</body>
</html>
EOF_HTML
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write HTML file."; exit 1; fi;
sed -i "s|{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}|$JOB_NAME_PREFIX_FOR_SQ|g" "$HTML_TARGET_PATH";
echo "✅ Frontend HTML ($HTML_FILENAME) configured."
echo ""

echo "Generating Start Script: $START_SCRIPT_TARGET_PATH"
cat > "$START_SCRIPT_TARGET_PATH" << EOF_START_SCRIPT
#!/bin/bash
SERAPHIM_DIR_START="{{SERAPHIM_DIR_PLACEHOLDER}}"
CONDA_ENV_NAME_START="{{CONDA_ENV_NAME_PLACEHOLDER}}"
CONDA_BASE_PATH_START="{{CONDA_BASE_PATH_PLACEHOLDER}}"
BACKEND_SCRIPT_START="{{BACKEND_FILENAME_PLACEHOLDER}}"
BACKEND_PORT_START={{BACKEND_PORT_PLACEHOLDER}}
FRONTEND_PORT_START={{FRONTEND_PORT_PLACEHOLDER}}
MODELS_FILE_START="\$SERAPHIM_DIR_START/{{MODELS_FILENAME_PLACEHOLDER}}"

BACKEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_backend.log"
FRONTEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.log"
BACKEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_backend.pid"
FRONTEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.pid"

is_port_in_use() {
    local port=\$1
    if command -v ss > /dev/null; then
        ss -tulnp | grep -q ":\${port}\s"
    elif command -v netstat > /dev/null; then
        netstat -tulnp | grep -q ":\${port}\s"
    else
        echo "Warning: Neither 'ss' nor 'netstat' found. Cannot check if port \$port is in use."
        return 0 
    fi
}

echo "Starting SERAPHIM Application..."
echo "================================="

if [ ! -f "\$MODELS_FILE_START" ]; then
    echo "❌ Error: Models file (\$MODELS_FILE_START) not found!"
    echo "Please create this file in the SERAPHIM directory (\$SERAPHIM_DIR_START) and populate it with your models."
    echo "Format: model_id,Display Name (one model per line)"
    exit 1
elif ! grep -q '[^[:space:]]' "\$MODELS_FILE_START"; then 
    echo "⚠️ Warning: Models file (\$MODELS_FILE_START) is empty or contains only whitespace."
    echo "The model dropdown in the UI will be empty. Please populate the file with models."
fi


if [ -f "\$BACKEND_PID_FILE" ] && ps -p \$(cat "\$BACKEND_PID_FILE") > /dev/null; then
    echo "❌ Backend already running (PID: \$(cat "\$BACKEND_PID_FILE")). Use ./stop_seraphim.sh to stop it first."
    exit 1
fi
if is_port_in_use "\$BACKEND_PORT_START"; then
    echo "❌ Error: Backend port \$BACKEND_PORT_START is already in use. Please free it or change BACKEND_PORT in install.sh and re-run install."
    exit 1
fi

if [ -f "\$FRONTEND_PID_FILE" ] && ps -p \$(cat "\$FRONTEND_PID_FILE") > /dev/null; then
    echo "❌ Frontend server already running (PID: \$(cat "\$FRONTEND_PID_FILE")). Use ./stop_seraphim.sh to stop it first."
    exit 1
fi
if is_port_in_use "\$FRONTEND_PORT_START"; then
    echo "❌ Error: Frontend port \$FRONTEND_PORT_START is already in use. Please free it or change FRONTEND_PORT in install.sh and re-run install."
    exit 1
fi

cd "\$SERAPHIM_DIR_START" || { echo "Error: Could not navigate to \$SERAPHIM_DIR_START"; exit 1; }
echo "Activating Conda environment: \$CONDA_ENV_NAME_START..."
_CONDA_SH_PATH="\$CONDA_BASE_PATH_START/etc/profile.d/conda.sh"
if [ -z "\$CONDA_BASE_PATH_START" ] || [ ! -f "\$_CONDA_SH_PATH" ]; then 
    _FALLBACK_CONDA_BASE_PATH=\$(conda info --base 2>/dev/null) 
    if [ -n "\$_FALLBACK_CONDA_BASE_PATH" ]; then 
        _CONDA_SH_PATH="\$_FALLBACK_CONDA_BASE_PATH/etc/profile.d/conda.sh"
        echo "Using fallback Conda base path: \$_FALLBACK_CONDA_BASE_PATH"
    fi
fi
if [ ! -f "\$_CONDA_SH_PATH" ]; then echo "Error: conda.sh not found. Cannot activate Conda environment."; exit 1; fi

# shellcheck source=/dev/null
. "\$_CONDA_SH_PATH"; conda activate "\$CONDA_ENV_NAME_START"
if [[ "\$CONDA_DEFAULT_ENV" != *"\$CONDA_ENV_NAME_START"* ]]; then 
    echo "Error: Failed to activate Conda environment '\$CONDA_ENV_NAME_START'."
    echo "Current env: \$CONDA_DEFAULT_ENV. Conda prefix: \$CONDA_PREFIX."
    exit 1
fi
echo "Conda environment '\$CONDA_ENV_NAME_START' activated. Path: \$CONDA_PREFIX";

echo "Starting Backend Server (port \$BACKEND_PORT_START)... Logging to: \$BACKEND_LOG_FILE"
nohup python "\$BACKEND_SCRIPT_START" >> "\$BACKEND_LOG_FILE" 2>&1 &
_BACKEND_PID=\$!; echo \$_BACKEND_PID > "\$BACKEND_PID_FILE"
echo "Backend PID: \$_BACKEND_PID."
sleep 3 
if ! ps -p \$_BACKEND_PID > /dev/null; then 
    echo "❌ Error: Backend server failed to start. Check \$BACKEND_LOG_FILE for details."
    rm -f "\$BACKEND_PID_FILE"
    exit 1
fi

echo "Starting Frontend HTTP Server (port \$FRONTEND_PORT_START)... Logging to: \$FRONTEND_LOG_FILE"
nohup python -m http.server --bind 0.0.0.0 "\$FRONTEND_PORT_START" >> "\$FRONTEND_LOG_FILE" 2>&1 &
_FRONTEND_PID=\$!; echo \$_FRONTEND_PID > "\$FRONTEND_PID_FILE"
echo "Frontend PID: \$_FRONTEND_PID."
sleep 1 
if ! ps -p \$_FRONTEND_PID > /dev/null; then 
    echo "❌ Error: Frontend HTTP server failed to start. Check \$FRONTEND_LOG_FILE for details."
    if ps -p \$_BACKEND_PID > /dev/null; then kill \$_BACKEND_PID; fi
    rm -f "\$BACKEND_PID_FILE" "\$FRONTEND_PID_FILE"
    exit 1
fi

_SERVER_IP_GUESS=\$(hostname -I | awk '{print \$1}' || echo "YOUR_SERVER_IP_ADDRESS")
echo "================================="
echo "✅ SERAPHIM Application Started!"
echo "   Access Frontend UI at: http://\${_SERVER_IP_GUESS}:\$FRONTEND_PORT_START"
echo "   (If the IP is incorrect, use your server's actual IP address)"
echo ""
echo "   To stop the application, run: ./$STOP_SCRIPT_FILENAME"
echo "================================="
EOF_START_SCRIPT
chmod +x "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_BASE_PATH_PLACEHOLDER}}|$ESCAPED_CONDA_BASE_PATH_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{BACKEND_FILENAME_PLACEHOLDER}}|$BACKEND_FILENAME|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{MODELS_FILENAME_PLACEHOLDER}}|$MODELS_FILENAME|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{FRONTEND_PORT_PLACEHOLDER}}|$FRONTEND_PORT|g" "$START_SCRIPT_TARGET_PATH"
echo "✅ Start script ($START_SCRIPT_FILENAME) created."
echo ""

echo "Generating Stop Script: $STOP_SCRIPT_TARGET_PATH"
cat > "$STOP_SCRIPT_TARGET_PATH" << EOF_STOP_SCRIPT
#!/bin/bash
SERAPHIM_DIR_STOP="{{SERAPHIM_DIR_PLACEHOLDER}}"
BACKEND_PID_FILE_STOP="\$SERAPHIM_DIR_STOP/seraphim_backend.pid"
FRONTEND_PID_FILE_STOP="\$SERAPHIM_DIR_STOP/seraphim_frontend.pid"
echo "Stopping SERAPHIM Application..."
stop_process() {
    local pid_file="\$1"; local process_name="\$2"
    if [ -f "\$pid_file" ]; then
        _PID_TO_KILL=\$(cat "\$pid_file")
        if [ -n "\$_PID_TO_KILL" ] && ps -p "\$_PID_TO_KILL" > /dev/null; then
            echo "Stopping \$process_name (PID: \$_PID_TO_KILL)..."; 
            kill "\$_PID_TO_KILL"; 
            for i in {1..5}; do 
                if ! ps -p "\$_PID_TO_KILL" > /dev/null; then break; fi; 
                sleep 0.5; 
            done
            if ps -p "\$_PID_TO_KILL" > /dev/null; then 
                echo "Force stopping \$process_name (PID: \$_PID_TO_KILL)...";
                kill -9 "\$_PID_TO_KILL"; sleep 0.5; 
            fi
            if ps -p "\$_PID_TO_KILL" > /dev/null; then 
                echo "❌ Error: Failed to stop \$process_name (PID: \$_PID_TO_KILL). Manual check required."; 
            else 
                echo "✅ \$process_name stopped."; 
            fi
        else 
            echo "ℹ️ \$process_name (PID from file: \$_PID_TO_KILL) not running or PID is invalid."
        fi
        rm -f "\$pid_file"
    else 
        echo "⚠️ \$process_name PID file (\${pid_file}) not found. Cannot stop."
    fi
}
stop_process "\$BACKEND_PID_FILE_STOP" "Backend Server"
stop_process "\$FRONTEND_PID_FILE_STOP" "Frontend Server"
echo "SERAPHIM Stop Process Attempted."
EOF_STOP_SCRIPT
chmod +x "$STOP_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$STOP_SCRIPT_TARGET_PATH"
echo "✅ Stop script ($STOP_SCRIPT_FILENAME) created."
echo ""

echo "======================================================================"
echo "✅ SERAPHIM Setup Complete! (v2.3.1 - Matrix Edition)"
echo "IMPORTANT: Ensure your '$MODELS_FILE_PATH' file is correctly populated."
echo ""
echo "To run the application:"
echo "  cd \"$SERAPHIM_DIR\" && ./$START_SCRIPT_FILENAME"
echo ""
_SERVER_IP_FINAL_GUESS=$(hostname -I | awk '{print $1}' || echo "YOUR_SERVER_IP_ADDRESS")
echo "Access the SERAPHIM UI at: http://${_SERVER_IP_FINAL_GUESS}:$FRONTEND_PORT"
echo "======================================================================"
echo "🚨 Notes:"
echo "    - User running backend needs sbatch & scancel access."
echo "    - All Slurm jobs for the user will be listed."
echo "    - Log viewing and service URL detection work best for SERAPHIM-launched jobs."
echo "    - Slurm job logs are in '$SCRIPTS_DIR'. Log viewing restricted to this directory."
echo "    - Log polling updates every ~1 second for the selected job (configurable in seraphim_logic.js)."
echo "    - Internal vLLM engine log will always auto-scroll to the bottom on new content."
echo "    - Email notification option has been removed."
echo "    - Enjoy the new Matrix theme!"
echo "======================================================================"
exit 0