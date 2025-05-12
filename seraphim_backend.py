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

SERAPHIM_DIR_PY = "/home/aimotion_api/SERAPHIM"
SCRIPTS_DIR_PY = "/home/aimotion_api/SERAPHIM/scripts"
VLLM_LOG_DIR_PY = "/home/aimotion_api/SERAPHIM/seraphim_internal_logs"
CONDA_ENV_NAME_PY = "seraphim_vllm_env"
BACKEND_PORT_PY = 8870
JOB_NAME_PREFIX_FOR_SQ_PY = "vllm_service" # For default job name creation

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
