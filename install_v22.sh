#!/bin/bash

# SERAPHIM Installation Script - v2.2
# 4-Col Layout, Green Log Text, Faster Log Polling, Updated Cancel Dialog,
# New Log Titles, Tracks All User Slurm Jobs, Visually Integrated Model Search.
# Auto-select new job and auto-display its logs.
# Assumes models.txt, Searchable Dropdown, Port Checks, Active Deployments, User-defined max_model_len
# Includes VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for overriding model length checks.
# vLLM service runs in FOREGROUND within Slurm job.
# Job Cancellation and Log Viewing capabilities.

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
JOB_NAME_PREFIX_FOR_SQ="vllm_service" # Still used for default job naming, but not for filtering display

if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found."
    exit 1
fi

echo "Starting SERAPHIM vLLM Deployment Setup (v2.2)..."
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
import re # For parsing squeue and log files
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
JOB_NAME_PREFIX_FOR_SQ_PY = "{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}" # For default job name creation

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
    mail_user: str | None = None
    max_model_len: Optional[int] = Field(None, gt=0)

class DeployedServiceInfo(BaseModel):
    job_id: str
    job_name: str
    status: str
    nodes: Optional[str] = None
    partition: Optional[str] = None
    time_used: Optional[str] = None
    user: Optional[str] = None
    service_url: Optional[str] = None
    slurm_output_file: Optional[str] = None
    slurm_error_file: Optional[str] = None
    raw_squeue_line: Optional[str] = None


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
        logger.info(f"Using user-defined max_model_len: {current_max_model_len} for {config.selected_model}")
    else:
        if "llama-2-7b" in config.selected_model.lower() or "llama2-7b" in config.selected_model.lower():
            current_max_model_len = 4096
            logger.info(f"Defaulted max_model_len to {current_max_model_len} for {config.selected_model} (no user input)")
        elif "mixtral" in config.selected_model.lower() or "pixtral" in config.selected_model.lower():
            current_max_model_len = 32768
            logger.info(f"Defaulted max_model_len to {current_max_model_len} for {config.selected_model} (no user input)")
        else: 
            logger.info(f"Using general default max_model_len: {current_max_model_len} for {config.selected_model} (no user input or specific rule)")
            
    if "pixtral" in config.selected_model.lower():
        model_args.append('--guided-decoding-backend=lm-format-enforcer')
        model_args.append("--limit_mm_per_prompt 'image=8'")
        if "mistralai/Pixtral-12B-2409" in config.selected_model:
            model_args.extend(['--enable-auto-tool-choice', '--tool-call-parser=mistral',
                               '--tokenizer_mode mistral', '--revision aaef4baf771761a81ba89465a18e4427f3a105f9'])
                               
    model_args.append(f'--max-model-len {current_max_model_len}')
    vllm_serve_command_full = vllm_serve_command + " \\\n    " + " \\\n    ".join(model_args)

    mail_type_line = f"#SBATCH --mail-type=ALL\\n#SBATCH --mail-user={config.mail_user}" if config.mail_user else "#SBATCH --mail-type=NONE"
    safe_filename_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config.job_name)
    slurm_job_name = config.job_name # Use the user-provided job name for #SBATCH --job-name
    unique_id = str(uuid.uuid4())[:8]
    script_filename = f"deploy_{safe_filename_job_name}_{unique_id}.slurm" # Use safe name for script file
    script_path = os.path.join(scripts_dir, script_filename)

    # Use safe_filename_job_name for the log file patterns, consistent with how it's derived in get_active_deployments
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

def parse_slurm_log_for_url(log_file_path: str, job_node: str, job_port: str) -> Optional[str]:
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
                        service_host = job_node if job_node and log_ip == "0.0.0.0" else log_ip
                        if service_host:
                            return f"http://{service_host}:{job_port}/docs"
            logger.debug(f"Uvicorn URL line not found in first 200 lines of {log_file_path}")
            if job_node and job_port: 
                return f"http://{job_node}:{job_port}/docs (best guess, check log for actual URL)"
    except Exception as e:
        logger.warning(f"Could not read or parse log file {log_file_path}: {e}")
    return None

@app.get("/api/active_deployments", response_model=List[DeployedServiceInfo])
async def get_active_deployments():
    deployments = []
    try:
        user = subprocess.check_output("whoami", text=True).strip()
        squeue_cmd = ["squeue", "-u", user, "-o", "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R",
                      "--noheader", "--states=RUNNING,PENDING,COMPLETING,SUSPENDED"] # Added more states
        process = subprocess.run(squeue_cmd, capture_output=True, text=True, check=True, timeout=15)
        
        for line in process.stdout.strip().split('\n'):
            if not line.strip(): continue
            try:
                parts = line.split(maxsplit=7) 
                if len(parts) < 8:
                    logger.warning(f"Skipping malformed squeue line (not enough parts): {line}")
                    continue

                job_id = parts[0].strip()
                partition = parts[1].strip()
                job_name_squeue = parts[2].strip() 
                user_val = parts[3].strip()
                state_val = parts[4].strip()
                time_val = parts[5].strip()
                node_list_val = parts[7].strip()   

                # REMOVED FILTER: if not job_name_squeue.startswith(JOB_NAME_PREFIX_FOR_SQ_PY): continue

                # Log files are named based on the SLURM job name (#SBATCH --job-name=...).
                # This job_name_squeue is that name.
                safe_job_name_for_logs = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in job_name_squeue)
                slurm_out_file_path = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name_for_logs}_{job_id}.out")
                slurm_err_file_path = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name_for_logs}_{job_id}.err")
                
                service_info = DeployedServiceInfo(
                    job_id=job_id, job_name=job_name_squeue, status=state_val,
                    nodes=node_list_val if state_val in ["R", "RUNNING"] and node_list_val and node_list_val != "(None)" else None,
                    partition=partition, time_used=time_val, user=user_val,
                    slurm_output_file=slurm_out_file_path, 
                    slurm_error_file=slurm_err_file_path,   
                    raw_squeue_line=line
                )

                if service_info.status in ["R", "RUNNING"] and service_info.nodes:
                    job_port_from_name = "8000" # Default, might not be relevant for non-SERAPHIM jobs
                    # Attempt to extract port only if it looks like a SERAPHIM job
                    if job_name_squeue.startswith(JOB_NAME_PREFIX_FOR_SQ_PY):
                        port_match_in_name = re.search(r"_p(\d{4,5})", job_name_squeue) 
                        if port_match_in_name:
                            job_port_from_name = port_match_in_name.group(1)
                    
                    if os.path.exists(slurm_out_file_path): # Only parse if log exists
                        service_info.service_url = parse_slurm_log_for_url(
                            slurm_out_file_path,
                            service_info.nodes.split(',')[0].split('(')[0].strip(), 
                            job_port_from_name
                        )
                
                deployments.append(service_info)
            except Exception as e:
                logger.error(f"Error parsing squeue line '{line}': {e}", exc_info=True)
                deployments.append(DeployedServiceInfo(job_id="PARSE_ERROR", job_name=f"Error processing: {line[:60]}...", status="UNKNOWN"))
    except subprocess.TimeoutExpired:
        logger.error("squeue command timed out.")
    except subprocess.CalledProcessError as e:
        logger.error(f"squeue command failed with RC {e.returncode}: {e.stderr}")
    except Exception as e:
        logger.error(f"Error fetching active deployments: {e}", exc_info=True)
    return deployments

@app.post("/api/cancel_job/{job_id}")
async def cancel_slurm_job_api(job_id: str):
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
            if process.stdout.strip(): error_detail += f" Stdout: {process.stdout.strip()}"
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
        # Max lines to read from different parts of the file
        max_lines_total = 300 # Reduced max lines for more frequent polling
        max_bytes_head = 30 * 1024 
        max_bytes_tail = 70 * 1024 
        
        if file_size == 0:
            return {"log_content": "(Log file is empty)"}

        if file_size <= (max_bytes_head + max_bytes_tail): 
             with open(abs_file_path, 'r', errors='ignore') as f:
                log_content_lines = [line for i, line in enumerate(f) if i < max_lines_total]
                if len(log_content_lines) == max_lines_total:
                    current_pos = f.tell()
                    f.seek(0, os.SEEK_END)
                    end_pos = f.tell()
                    if current_pos < end_pos : # Check if there's more content
                        log_content_lines.append(f"\n--- (Log truncated, showing first {max_lines_total} lines from small file) ---\n")
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
            
            log_content_lines.append(f"\n\n... (log truncated - {file_size // 1024} KB total, showing ~{head_lines_count} head and ~{tail_lines_count} tail lines) ...\n\n")
            
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
let currentSelectedJobDetails = null; // Stores { jobId, outFile, errFile } for polling
// Adjusted LOG_REFRESH_INTERVAL to 1 second (1000ms). 500ms is very frequent for file I/O.
// You can change it to 500 if you are sure about the server load capacity.
const LOG_REFRESH_INTERVAL = 1000; 

async function fetchAndPopulateModels() {
    console.log("SERAPHIM_DEBUG: Fetching models from", MODELS_FILE_URL);
    const modelSelect = document.getElementById('model-select');
    const modelSearchInput = document.getElementById('model-search'); // Keep the input for filtering logic
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
        populateModelDropdown(allModels); // Initial population
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        modelSelect.innerHTML = `<option value="">-- Error loading models --</option>`;
        if(document.getElementById('output')) document.getElementById('output').textContent = `❌ ${error.message}`;
    }
}

function populateModelDropdown(modelsToDisplay) {
    const modelSelect = document.getElementById('model-select');
    const currentSearchVal = document.getElementById('model-search').value; // Preserve current selection if possible
    const currentSelectedVal = modelSelect.value;

    modelSelect.innerHTML = ''; // Clear existing options
    // Add a default placeholder if no models match or for initial state
    if (modelsToDisplay.length === 0 && currentSearchVal === "") {
        const placeholder = document.createElement('option');
        placeholder.value = ""; 
        placeholder.textContent = "-- Select a Model --";
        modelSelect.appendChild(placeholder);
    } else if (modelsToDisplay.length === 0 && currentSearchVal !== "") {
         const noMatch = document.createElement('option');
        noMatch.value = ""; 
        noMatch.textContent = "-- No models match search --";
        modelSelect.appendChild(noMatch);
    }


    modelsToDisplay.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id; 
        option.textContent = model.name;
        modelSelect.appendChild(option);
    });

    // Try to reselect previous value if it's still in the filtered list
    if (modelsToDisplay.some(m => m.id === currentSelectedVal)) {
        modelSelect.value = currentSelectedVal;
    } else if (modelsToDisplay.length > 0 && currentSearchVal === "") {
        // If search is cleared and previous selection is gone, maybe select the first? Or default.
        // For now, let it default to the first in the list or blank.
    }
     if(modelSelect.options.length > 0 && !modelSelect.value && currentSearchVal === ""){
        // if nothing is selected and search is empty, add placeholder
        const placeholder = document.createElement('option');
        placeholder.value = ""; 
        placeholder.textContent = "-- Select a Model --";
        modelSelect.insertBefore(placeholder, modelSelect.firstChild);
        modelSelect.value = "";
    }
}

function filterModels() {
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
                         displayElement.textContent.startsWith('Newly submitted') ||
                         displayElement.textContent.startsWith('Selected job');

    if (initialFetch) {
      displayElement.textContent = `🔄 Fetching ${filePath.split('/').pop()}...`;
    }
    
    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/log_content?file_path=${encodeURIComponent(filePath)}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        
        const newContent = result.log_content || '(empty log file)';
        if (displayElement.textContent !== newContent) { // Only update DOM if content changed
             displayElement.textContent = newContent;
             // Scroll to bottom only if it was an initial fetch or content has significantly changed (heuristic)
             if (initialFetch || Math.abs(newContent.length - displayElement.textContent.length) > 100) {
                 displayElement.scrollTop = displayElement.scrollHeight;
             }
        }

    } catch (error) {
        console.error(`SERAPHIM_DEBUG: Error fetching log ${filePath}:`, error);
        // Avoid constant error spam if polling fails, only update if it was an initial fetch
        if (initialFetch) {
            displayElement.textContent = `❌ Error fetching log: ${error.message}`;
        }
    }
}

async function cancelJob(jobId) {
    const outputDiv = document.getElementById('output'); // For deploy form status messages
    // Updated confirmation message
    if (!confirm(`Are you sure you want to cancel this job? (Be mindful that other users may be using it, consult with your colleagues before proceeding)`)) {
        return;
    }
    outputDiv.textContent = `🔄 Attempting to cancel job ${jobId}...`;
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
    console.log("SERAPHIM_DEBUG: Refreshing active deployments...");
    const listDiv = document.getElementById('deployed-endpoints-list');
    const refreshButton = document.getElementById('refresh-endpoints-button');
    const logOutDisplay = document.getElementById('log-output-content');
    const logErrDisplay = document.getElementById('log-error-content');

    if (!listDiv || !refreshButton || !logOutDisplay || !logErrDisplay) return;

    listDiv.innerHTML = "<p><em>🔄 Fetching active deployments...</em></p>";
    // Clear current selection for polling *unless* we are trying to auto-select this specific job again
    if (!jobToSelect || (jobToSelect && currentSelectedJobDetails && jobToSelect.jobId !== currentSelectedJobDetails.jobId)) {
        if (!jobToSelect) { // General refresh, not an auto-select action
            currentSelectedJobDetails = null;
            logOutDisplay.textContent = "Select a job to view its log.";
            logErrDisplay.textContent = "Select a job to view its log.";
        }
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
                    ${job.nodes ? `<br/><strong>Node(s):</strong> ${job.nodes}` : (job.status === 'PD' ? '<br/><em>Pending Allocation</em>' : '')}
                    ${job.service_url ? `<br/><strong>URL:</strong> <a href="${job.service_url}" target="_blank" onclick="event.stopPropagation();">${job.service_url}</a>` : ''}
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
            // If no job was auto-selected (either not requested or not found), and a job was previously selected,
            // check if that previously selected job still exists in the new list. If not, clear selection.
            if (!jobAutoSelected && currentSelectedJobDetails) {
                const stillExists = deployments.some(d => d.job_id === currentSelectedJobDetails.jobId);
                if (!stillExists) {
                    currentSelectedJobDetails = null;
                    logOutDisplay.textContent = "Previously selected job no longer active.";
                    logErrDisplay.textContent = "";
                } else { // Previously selected job still exists, re-highlight it
                    const itemToReselect = listDiv.querySelector(`.endpoint-item[data-jobid="${currentSelectedJobDetails.jobId}"]`);
                    if (itemToReselect) itemToReselect.classList.add('selected');
                }
            } else if (!jobAutoSelected && !currentSelectedJobDetails && deployments.length > 0) {
                 // If nothing is selected, clear the log panes.
                 logOutDisplay.textContent = "Select a job to view its log.";
                 logErrDisplay.textContent = "Select a job to view its log.";
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
        mail_user: document.getElementById('mail-user').value || null,
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
    await fetchAndPopulateModels(); // Populates allModels and initial dropdown
    
    document.getElementById('model-search')?.addEventListener('input', filterModels); // Search input filters the select
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
            --primary-color: #029702; --secondary-color: #f3cb00; --accent-color: #ffcc00;
            --bg-color: #1c1c1e; --card-bg-color: #2c2c2e; --text-color: #e5e5e7;
            --text-muted-color: #8e8e93; --border-color: #3a3a3c;
            --font-body: 'Exo 2', sans-serif; --font-heading: 'Orbitron', sans-serif;
            --success-color: #02c702; --warning-color: #f3cb00; --error-color: #ff3b30;
            --cancel-button-bg: #d9534f; --cancel-button-hover-bg: #c9302c;
            --log-text-color: var(--success-color); /* Green for log text */
        }
        html, body { height: 100%; margin: 0; padding: 0; overflow-x: hidden; }
        body { font-family: var(--font-body); background-color: var(--bg-color); color: var(--text-color); display: flex; flex-direction: column; font-size: 16px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); color: white; padding: 15px 30px; text-align: center; border-bottom: 3px solid var(--accent-color); text-shadow: 0 1px 3px rgba(0,0,0,0.3); flex-shrink: 0; }
        .header h1 { margin: 0; font-family: var(--font-heading); font-size: 2.1em; font-weight: 700; letter-spacing: 1.5px; display: flex; align-items: center; justify-content: center; gap: 12px; }
        .header p { margin: 5px 0 0; font-size: 0.9em; opacity: 0.9; font-weight: 300;}
        
        .page-content-wrapper {
            display: flex;
            flex-grow: 1; 
            padding: 15px;
            gap: 15px;    
            overflow: hidden; 
            max-width: 100%; 
            box-sizing: border-box;
            align-items: stretch; 
        }

        .column { 
            display: flex;
            flex-direction: column; 
            background-color: var(--card-bg-color);
            padding: 15px; 
            border-radius: 10px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            overflow: hidden; 
        }
        
        .log-column { flex: 1 1 22%; min-width: 280px; } 
        .deploy-column { flex: 1.5 1 28%; min-width: 380px; } 
        .endpoints-column { flex: 1.3 1 28%; min-width: 320px; }


        .column h3 {
            font-family: var(--font-heading); color: var(--secondary-color); 
            border-bottom: 1px solid var(--accent-color); padding-bottom: 8px; 
            margin-top: 0; margin-bottom: 15px; font-size: 1.2em; 
            letter-spacing: 1px; 
            display: flex; align-items: center; gap: 10px; flex-shrink: 0;
        }
        
        .form-container, .endpoints-container-inner {
             display: flex; flex-direction: column;
             overflow: hidden; 
             flex-grow: 1; 
        }
        
        .log-column pre { 
            background-color: #1a1a1c; 
            color: var(--log-text-color); 
            padding: 10px; 
            border-radius: 5px; font-family: 'Courier New', Courier, monospace;
            font-size: 0.8em; 
            white-space: pre-wrap; word-wrap: break-word;
            border: 1px solid #333;
            flex-grow: 1; 
            overflow-y: auto; 
            min-height: 150px; /* Ensure it has some base height */
        }
        
        /* Searchable Select Container */
        .searchable-select-container {
            display: flex;
            flex-direction: column;
            margin-bottom: 10px; /* Spacing after the whole unit */
        }
        #model-search {
            width: 100%; padding: 8px; 
            box-sizing: border-box; font-size: 0.9em; 
            background-color: #3a3a3c; color: var(--text-color); 
            border: 1px solid var(--border-color);
            border-bottom: none; /* Connect to select below */
            border-radius: 5px 5px 0 0; /* Top corners rounded */
            margin-bottom: 0; /* Remove default margin */
        }
        #model-search:focus {
             border-color: var(--primary-color); 
             box-shadow: 0 0 0 2px rgba(10, 132, 255, 0.25); 
             outline: none; background-color: #4a4a4e; 
             z-index: 10; /* Ensure focus ring is visible */
        }
        #model-select {
            border-radius: 0 0 5px 5px; /* Bottom corners rounded */
            margin-top: 0; /* Remove default margin */
            /* Ensure select takes focus styling when search is focused if desired (hard with pure CSS) */
        }
        /* End Searchable Select Container */


        label { display: block; margin-top: 10px; margin-bottom: 3px; font-weight: 600; font-size: 0.8em; color: var(--text-muted-color); text-transform: uppercase; letter-spacing: 0.5px;}
        /* General input/select styling (excluding model-search which is handled above) */
        select:not(#model-select), input[type="text"]:not(#model-search), input[type="number"], input[type="email"], input[type="password"] { 
            width: 100%; padding: 8px; margin-bottom: 10px; 
            border-radius: 5px; border: 1px solid var(--border-color); 
            box-sizing: border-box; font-size: 0.9em; 
            background-color: #3a3a3c; color: var(--text-color); 
        }
        select:not(#model-select):focus, input[type="text"]:not(#model-search):focus, input[type="number"]:focus, input[type="email"]:focus, input[type="password"]:focus { 
            border-color: var(--primary-color); 
            box-shadow: 0 0 0 2px rgba(10, 132, 255, 0.25); 
            outline: none; background-color: #4a4a4e; 
        }

        button { 
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color)); 
            color: white; padding: 10px 15px; cursor: pointer; border: none; 
            border-radius: 5px; font-weight: bold; font-size: 0.9em; 
            text-transform: uppercase; letter-spacing: 0.8px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); 
            transition: all 0.3s ease; margin-top: 8px; width: 100%; 
        }
        button:hover:not(:disabled), button:focus:not(:disabled) { 
            background: linear-gradient(to right, var(--secondary-color), var(--primary-color)); 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3); transform: translateY(-1px); 
        }
        button:disabled { background: #555; cursor: not-allowed; opacity: 0.7; }
        
        #output { 
            margin-top: 10px; padding: 10px; background-color: #1c1c1e; 
            border: 1px solid var(--border-color); border-radius: 6px; 
            white-space: pre-wrap; word-wrap: break-word; font-size: 0.8em; 
            max-height: 100px; overflow-y: auto; line-height: 1.4; 
            font-family: 'Courier New', Courier, monospace; color: var(--text-color);
            flex-shrink: 0; 
        }
        .slurm-options h3 { margin-top: 15px; font-size: 1em;}
        .slurm-options label { font-size: 0.75em; margin-top: 5px; }
        
        #refresh-endpoints-button { background: linear-gradient(to right, #ffcc00, #ff9500); margin-bottom: 10px; flex-shrink:0; }
        #refresh-endpoints-button:hover:not(:disabled) { background: linear-gradient(to right, #ff9500, #ffcc00); }
        
        #deployed-endpoints-list { overflow-y: auto; flex-grow: 1; padding-right: 5px; /* space for scrollbar */ }
        #deployed-endpoints-list ul { list-style-type: none; padding: 0; margin:0;}
        .endpoint-item { 
            background-color: #3a3a3cdd; border: 1px solid #4a4a4e; 
            padding: 8px; margin-bottom: 6px; border-radius: 6px; 
            font-size: 0.75em; line-height: 1.3; 
            cursor: pointer; transition: background-color 0.2s ease; 
        }
        .endpoint-item:hover { background-color: #4f4f52; }
        .endpoint-item.selected { background-color: var(--primary-color); color: white; border-left: 4px solid var(--secondary-color); }
        .endpoint-item.selected strong { color: var(--secondary-color); } 
        .endpoint-item.selected a { color: white; }
        .endpoint-item strong { color: var(--secondary-color); }
        .endpoint-item a { color: var(--accent-color); text-decoration: none; font-weight: bold; word-break: break-all;}
        .endpoint-item a:hover { text-decoration: underline; color: #ffd633; }
        .cancel-job-button {
            background: var(--cancel-button-bg); color: white;
            padding: 3px 6px; font-size: 0.9em; 
            margin-top: 5px; width: auto; display: inline-block; float: right;
        }
        .cancel-job-button:hover:not(:disabled) { background: var(--cancel-button-hover-bg); }

        .footer { text-align: center; padding: 15px; background-color: #0e0e0f; color: #8e8e93; font-size: 0.85em; border-top: 3px solid var(--accent-color); flex-shrink: 0;}
        .icon { margin-right: 8px; font-size: 1.1em; vertical-align: middle;}
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="icon">⚡</span> SERAPHIM <span class="icon">⚡</span></h1>
        <p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p>
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
                    <input type="text" id="model-search" placeholder="Type to filter models..." autocomplete="off">
                    <select id="model-select"><option value="">-- Loading models... --</option></select>
                </div>
                
                <label for="max-model-len">Max Model Length (Optional):</label>
                <input type="number" id="max-model-len" placeholder="e.g., 4096 (blank for default)" min="1"/>
                <label for="service-port">Service Port (on Slurm node):</label>
                <input type="number" id="service-port" value="8000" min="1024" max="65535"/>
                <label for="hf-token">Hugging Face Token (Optional):</label>
                <input type="password" id="hf-token" placeholder="For Llama, gated models, etc."/>
                
                <div class="slurm-options">
                    <h3>Slurm Configuration</h3>
                    <label for="job-name">Job Name (Prefix '{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}'...):</label>
                    <input type="text" id="job-name" value="{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}_model_port"/>
                    <label for="time-limit">Time Limit (HH:MM:SS):</label><input type="text" id="time-limit" value="23:59:59"/>
                    <label for="gpus">GPUs (e.g., 1 or a100:1):</label><input type="text" id="gpus" value="1"/>
                    <label for="cpus-per-task">CPUs per Task:</label><input type="number" id="cpus-per-task" value="4" min="1"/>
                    <label for="mem">Memory (e.g., 32G):</label><input type="text" id="mem" value="32G"/>
                    <label for="mail-user">Email Notify (Optional):</label><input type="email" id="mail-user" placeholder="your_email@example.com"/>
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

    <div class="footer">✧ SERAPHIM CORE Interface v2.2 (ANDERSON DE LIMA LUIZ) ✧</div>
    <script src="seraphim_logic.js" defer></script>
</body>
</html>
EOF_HTML
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write HTML file."; exit 1; fi;
sed -i "s|{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}|$JOB_NAME_PREFIX_FOR_SQ|g" "$HTML_TARGET_PATH";
echo "✅ Frontend HTML ($HTML_FILENAME) configured."
echo ""

# Start and Stop scripts remain unchanged from v2.0/v2.1 but are included for completeness
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
echo "✅ SERAPHIM Setup Complete! (v2.2)"
echo "IMPORTANT: Ensure your '$MODELS_FILE_PATH' file is correctly populated."
echo ""
echo "To run the application:"
echo "  cd \"$SERAPHIM_DIR\" && ./$START_SCRIPT_FILENAME"
echo ""
_SERVER_IP_FINAL_GUESS=$(hostname -I | awk '{print $1}' || echo "YOUR_SERVER_IP_ADDRESS")
echo "Access the SERAPHIM UI at: http://${_SERVER_IP_FINAL_GUESS}:$FRONTEND_PORT"
echo "======================================================================"
echo "🚨 Notes:"
echo "   - User running backend needs sbatch & scancel access."
echo "   - All Slurm jobs for the user will be listed."
echo "   - Log viewing and service URL detection work best for SERAPHIM-launched jobs."
echo "   - Slurm job logs are in '$SCRIPTS_DIR'. Log viewing restricted to this directory."
echo "   - Log polling updates every ~1 second for the selected job."
echo "   - Model search input is now visually integrated with the select dropdown."
echo "======================================================================"
exit 0
