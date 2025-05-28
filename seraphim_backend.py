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

SERAPHIM_DIR_PY = "/home/aimotion_api/SERAPHIM"
SCRIPTS_DIR_PY = "/home/aimotion_api/SERAPHIM/scripts"
VLLM_LOG_DIR_PY = "/home/aimotion_api/SERAPHIM/seraphim_internal_logs"
CONDA_ENV_NAME_PY = "seraphim_vllm_env"
BACKEND_PORT_PY = 8870
JOB_NAME_PREFIX_FOR_SQ_PY = "vllm_service" 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"],
)

class SlurmConfig(BaseModel):
    selected_model: str
    service_port: str
    hf_token: Optional[str] = None 
    job_name: str
    time_limit: str
    gpus: str
    cpus_per_task: str
    mem: str
    max_model_len: Optional[int] = Field(None, gt=0)

class DeployedServiceInfo(BaseModel):
    job_id: str
    job_name: str
    status: str
    nodes: Optional[str] = None
    node_ip: Optional[str] = None 
    partition: Optional[str] = None
    time_used: Optional[str] = None
    user: Optional[str] = None
    service_url: Optional[str] = None
    detected_port: Optional[str] = None
    slurm_output_file: Optional[str] = None
    slurm_error_file: Optional[str] = None
    raw_squeue_line: Optional[str] = None

def get_ip_from_node_name(node_name: Optional[str]) -> Optional[str]:
    if not node_name:
        return None
    node_name_clean = node_name.split('(')[0].strip()
    match_ki_g = re.fullmatch(r"ki-g(\d+)", node_name_clean)
    if match_ki_g:
        numeric_part = int(match_ki_g.group(1))
        return f"10.16.246.{numeric_part}"
    logger.debug(f"Node name '{node_name_clean}' did not match specific IP patterns. Will use nodename for URL if needed.")
    return node_name_clean

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
    elif "llama-2-7b" in config.selected_model.lower() or "llama2-7b" in config.selected_model.lower(): current_max_model_len = 4096
    elif "mixtral" in config.selected_model.lower() or "pixtral" in config.selected_model.lower(): current_max_model_len = 32768
    
    if "pixtral" in config.selected_model.lower():
        model_args.extend(['--guided-decoding-backend=lm-format-enforcer', "--limit_mm_per_prompt 'image=8'"])
        if "mistralai/Pixtral-12B-2409" in config.selected_model:
            model_args.extend(['--enable-auto-tool-choice', '--tool-call-parser=mistral',
                               '--tokenizer_mode mistral', '--revision aaef4baf771761a81ba89465a18e4427f3a105f9'])
                                
    model_args.append(f'--max-model-len {current_max_model_len}')
    vllm_serve_command_full = vllm_serve_command + " \\\n    " + " \\\n    ".join(model_args)

    mail_type_line = "#SBATCH --mail-type=NONE"

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
echo "Model Identifier: {config.selected_model}"
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
    except IOError as e: raise HTTPException(status_code=500, detail=f"Server error writing script: {e}")
    try:
        process = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True, timeout=30)
        job_id_message = process.stdout.strip()
        job_id = job_id_message.split(" ")[-1].strip() if "Submitted batch job" in job_id_message else "Unknown"
        actual_slurm_out = slurm_out_file_pattern.replace('%j', job_id if job_id != "Unknown" else "<JOB_ID>")
        actual_slurm_err = slurm_err_file_pattern.replace('%j', job_id if job_id != "Unknown" else "<JOB_ID>")
        return {"message": f"Slurm job submitted! ({job_id_message})", "job_id": job_id, "script_path": script_path, 
                "slurm_output_file_pattern": actual_slurm_out, "slurm_error_file_pattern": actual_slurm_err,  
                "monitoring_note": f"Monitor Slurm output ({actual_slurm_out}) for service logs."}
    except subprocess.TimeoutExpired: raise HTTPException(status_code=500, detail="sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Sbatch failed. RC: {e.returncode}. Stderr: {e.stderr.strip() if e.stderr else 'N/A'}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Unexpected sbatch error: {str(e)}")


def parse_slurm_log_for_url(log_file_path: str, host_identifier: str, job_port: str) -> Optional[str]:
    if not os.path.exists(log_file_path): return None
    try:
        with open(log_file_path, 'r', errors='ignore') as f:
            for _ in range(200): 
                line = f.readline()
                if not line: break
                match = re.search(r"Uvicorn running on http://([\d\.]+):(\d+)", line)
                if match:
                    log_ip, log_port_str = match.groups()
                    if log_port_str == job_port:
                        service_host = host_identifier if host_identifier and log_ip == "0.0.0.0" else log_ip
                        if service_host: return f"http://{service_host}:{job_port}" 
            if host_identifier and job_port:
                logger.info(f"Uvicorn line not found, constructing fallback URL for {host_identifier}:{job_port}")
                return f"http://{host_identifier}:{job_port}"
    except Exception as e: logger.warning(f"Could not read/parse log {log_file_path}: {e}")
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

                job_id, partition, job_name_squeue, user_val, state_val, time_val, _, node_list_raw = parts
                job_id = job_id.strip()
                partition = partition.strip()
                job_name_squeue = job_name_squeue.strip()
                user_val = user_val.strip()
                state_val = state_val.strip()
                time_val = time_val.strip()
                node_list_raw = node_list_raw.strip()

                safe_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in job_name_squeue)
                out_file = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name}_{job_id}.out")
                err_file = os.path.join(SCRIPTS_DIR_PY, f"{safe_job_name}_{job_id}.err")
                
                node_ip, first_node_name, detected_port, service_url = None, None, None, None

                if state_val in ["R", "RUNNING"] and node_list_raw and node_list_raw != "(None)":
                    first_node_name = node_list_raw.split(',')[0].strip()
                    node_ip = get_ip_from_node_name(first_node_name) 
                    
                    if job_name_squeue.startswith(JOB_NAME_PREFIX_FOR_SQ_PY):
                        port_match = re.search(r"_p(\d{4,5})", job_name_squeue)
                        if port_match: detected_port = port_match.group(1)
                    
                    if not detected_port: 
                        detected_port = "8000" 

                    host_identifier_for_url = node_ip or first_node_name 
                    if os.path.exists(out_file) and host_identifier_for_url and detected_port:
                        service_url = parse_slurm_log_for_url(out_file, host_identifier_for_url, detected_port)

                deployments.append(DeployedServiceInfo(
                    job_id=job_id, job_name=job_name_squeue, status=state_val,
                    nodes=node_list_raw if node_list_raw != "(None)" else None, node_ip=node_ip,
                    partition=partition, time_used=time_val, user=user_val,
                    service_url=service_url, detected_port=detected_port,
                    slurm_output_file=out_file, slurm_error_file=err_file, raw_squeue_line=line
                ))
            except Exception as e:
                logger.error(f"Error parsing squeue line '{line}': {e}", exc_info=True)
    except Exception as e: logger.error(f"Error fetching active deployments: {e}", exc_info=True)
    return deployments

@app.post("/api/cancel_job/{job_id}")
async def cancel_slurm_job_api(job_id: str):
    logger.info(f"Request to cancel Slurm job: {job_id}")
    try:
        if not re.match(r"^\d+$", job_id): raise HTTPException(status_code=400, detail="Invalid job ID format.")
        process = subprocess.run(["scancel", job_id], capture_output=True, text=True, timeout=10)
        if process.returncode == 0:
            return {"message": f"Cancellation sent for job {job_id}.", "details": process.stdout.strip()}
        else:
            err_detail = f"scancel failed for job {job_id} or job gone. RC:{process.returncode}."
            err_detail += f" E:{process.stderr.strip()}" if process.stderr else ""
            err_detail += f" O:{process.stdout.strip()}" if process.stdout else ""
            raise HTTPException(status_code=400, detail=err_detail)
    except subprocess.TimeoutExpired: raise HTTPException(status_code=500, detail=f"scancel for job {job_id} timed out.")
    except FileNotFoundError: raise HTTPException(status_code=500, detail="scancel command not found.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error cancelling job {job_id}: {e}")


@app.get("/api/log_content")
async def get_log_content_api(file_path: str = Query(...)):
    abs_scripts_dir = os.path.abspath(SCRIPTS_DIR_PY)
    abs_file_path = os.path.abspath(file_path)
    if not abs_file_path.startswith(abs_scripts_dir):
        raise HTTPException(status_code=403, detail="File access forbidden.")
    if not os.path.exists(abs_file_path):
        return {"log_content": f"(Log file not available: {os.path.basename(file_path)})"}
    if not os.path.isfile(abs_file_path):
        raise HTTPException(status_code=400, detail="Path is not a file.")

    try:
        file_size = os.path.getsize(abs_file_path)
        if file_size == 0: return {"log_content": "(Log file is empty)"}
        
        max_lines, head_kb, tail_kb = 300, 30, 70 
        
        content_lines = []
        if file_size <= (head_kb + tail_kb) * 1024:
            with open(abs_file_path, 'r', errors='ignore') as f:
                content_lines = [line for i, line in enumerate(f) if i < max_lines]
                if len(content_lines) == max_lines and f.readline(): 
                    content_lines.append(f"\n--- (Log truncated, showing first {max_lines} lines) ---\n")
        else:
            head_l_count = max_lines // 3
            tail_l_count = max_lines - head_l_count
            with open(abs_file_path, 'r', errors='ignore') as f:
                read_bytes = 0
                for i in range(head_l_count):
                    line = f.readline()
                    if not line or read_bytes > head_kb * 1024: break
                    content_lines.append(line)
                    read_bytes += len(line.encode('utf-8'))
            content_lines.append(f"\n\n... (log truncated - {file_size // 1024} KB total) ...\n\n")
            with open(abs_file_path, 'rb') as f:
                f.seek(max(0, file_size - tail_kb * 1024))
                if f.tell() > 0: f.readline() 
                raw_tail = [ln.decode('utf-8', errors='ignore') for ln in f.readlines()]
                content_lines.extend(raw_tail[-tail_l_count:])
        return {"log_content": "".join(content_lines)}
    except Exception as e: 
        logger.error(f"Error reading log {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading log: {e}")

if __name__ == "__main__":
    os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    os.makedirs(VLLM_LOG_DIR_PY, exist_ok=True) 
    logger.info(f"Starting SERAPHIM Backend Server on http://0.0.0.0:{BACKEND_PORT_PY}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT_PY)
