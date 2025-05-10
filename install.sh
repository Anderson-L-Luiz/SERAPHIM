#!/bin/bash

# SERAPHIM Installation Script - With Backend, Start & Stop Scripts
# vLLM service now runs in FOREGROUND within Slurm job.

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
CONDA_ENV_NAME="seraphim_vllm_env"
SERAPHIM_DIR="$HOME/SERAPHIM"
SCRIPTS_DIR="$SERAPHIM_DIR/scripts" # For Slurm scripts AND their output/error logs
VLLM_LOG_DIR_IN_INSTALLER="$SCRIPTS_DIR/vllm_service_specific_logs" # This will NOT be used by Slurm script directly for vLLM output anymore
HTML_FILENAME="seraphim_deploy.html"
JS_FILENAME="seraphim_logic.js"
BACKEND_FILENAME="seraphim_backend.py"
START_SCRIPT_FILENAME="start_seraphim.sh"
STOP_SCRIPT_FILENAME="stop_seraphim.sh"

HTML_TARGET_PATH="$SERAPHIM_DIR/$HTML_FILENAME"
JS_TARGET_PATH="$SERAPHIM_DIR/$JS_FILENAME"
BACKEND_TARGET_PATH="$SERAPHIM_DIR/$BACKEND_FILENAME"
START_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$START_SCRIPT_FILENAME"
STOP_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$STOP_SCRIPT_FILENAME"
VLLM_REQUIREMENTS_FILE="$SCRIPTS_DIR/vllm_requirements.txt"

BACKEND_PORT=8870
FRONTEND_PORT=8869

if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found."
    exit 1
fi

echo "Starting SERAPHIM vLLM Deployment Setup..."
echo "Target Directory: $SERAPHIM_DIR"
echo "=========================================================================="
mkdir -p "$SERAPHIM_DIR"
mkdir -p "$SCRIPTS_DIR"
# The VLLM_LOG_DIR_IN_INSTALLER is less critical now for vLLM output, but backend might use it for other logs if extended.
# For now, Slurm output/error files will be directly in SCRIPTS_DIR.
mkdir -p "$VLLM_LOG_DIR_IN_INSTALLER" 
echo "Directories checked/created."
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

SERAPHIM_DIR_PY = "{{SERAPHIM_DIR_PLACEHOLDER}}"
# SCRIPTS_DIR_PY is where Slurm scripts are saved AND where their .out/.err files will go.
SCRIPTS_DIR_PY = "{{SCRIPTS_DIR_PLACEHOLDER}}" 
# VLLM_LOG_DIR_PY is not directly used for vLLM output in Slurm script anymore.
# It could be used by the backend for its own general logging if desired.
VLLM_LOG_DIR_PY = "{{VLLM_LOG_DIR_PLACEHOLDER}}" 
CONDA_ENV_NAME_PY = "{{CONDA_ENV_NAME_PLACEHOLDER}}"
BACKEND_PORT_PY = {{BACKEND_PORT_PLACEHOLDER}}

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
PYTHON_EOF
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write Backend Python script."; exit 1; fi;
ESCAPED_SERAPHIM_DIR_FOR_SED=$(printf '%s\n' "$SERAPHIM_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_SCRIPTS_DIR_FOR_SED=$(printf '%s\n' "$SCRIPTS_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_VLLM_LOG_DIR_FOR_SED=$(printf '%s\n' "$VLLM_LOG_DIR_IN_INSTALLER" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_ENV_NAME_FOR_SED=$(printf '%s\n' "$CONDA_ENV_NAME" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_BASE_PATH_FOR_SED=$(printf '%s\n' "$CONDA_BASE_PATH" | sed 's:[&/\]:\\&:g')
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{SCRIPTS_DIR_PLACEHOLDER}}|$ESCAPED_SCRIPTS_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{VLLM_LOG_DIR_PLACEHOLDER}}|$ESCAPED_VLLM_LOG_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$BACKEND_TARGET_PATH"
echo "✅ Backend Python script ($BACKEND_FILENAME) configured."
echo ""

echo "Generating JavaScript logic file: $JS_TARGET_PATH";
cat > "$JS_TARGET_PATH" << 'EOF_JS'
// seraphim_logic.js
const BACKEND_API_URL = `http://${window.location.hostname}:{{BACKEND_PORT_PLACEHOLDER}}/api/deploy`;
const staticModels = [
    { id: "mistralai/Mistral-7B-Instruct-v0.1", name: "Mistral-7B-Instruct-v0.1" },
    { id: "meta-llama/Llama-2-7b-chat-hf", name: "Llama-2-7B-Chat-HF" },
    { id: "google/gemma-7b-it", name: "Gemma-7B-IT (Google)" },
    { id: "Qwen/Qwen2-7B-Instruct", name: "Qwen2-7B-Instruct" },
    { id: "BAAI/AquilaChat-7B", name: "AquilaChat-7B (BAAI)"},
    { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", name: "Mixtral-8x7B-Instruct-v0.1" },
    { id: "EleutherAI/gpt-j-6b", name: "GPT-J-6B (EleutherAI)"},
    { id: "tiiuae/falcon-7b-instruct", name: "Falcon-7B-Instruct (TII UAE)"},
    { id: "mistralai/Pixtral-12B-2409", name: "Pixtral-12B-2409 (Multimodal)" },
    { id: "microsoft/phi-2", name: "Phi-2 (Microsoft)"}
];

function populateStaticModels() {
    const modelSelect = document.getElementById('model-select');
    if (!modelSelect) { console.error("SERAPHIM_DEBUG: 'model-select' not found!"); return; }
    while (modelSelect.options.length > 1) { modelSelect.remove(modelSelect.options.length - 1); }
    staticModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id; option.textContent = model.name;
        modelSelect.appendChild(option);
    });
}

function refreshDeployedEndpoints() {
    const listDiv = document.getElementById('deployed-endpoints-list');
    if (listDiv) listDiv.innerHTML = "<p><em>Listing active jobs requires further backend integration.</em></p>";
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
        job_name: document.getElementById('job-name').value,
        time_limit: document.getElementById('time-limit').value,
        gpus: document.getElementById('gpus').value,
        cpus_per_task: document.getElementById('cpus-per-task').value,
        mem: document.getElementById('mem').value,
        mail_user: document.getElementById('mail-user').value || null,
    };

    if (!slurmConfig.selected_model) {
        outputDiv.textContent = "⚠️ Please select a model.";
        outputDiv.style.color = "var(--warning-color, #f3cb00)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend"; return;
    }
    // Add more client-side validation as needed

    try {
        const response = await fetch(BACKEND_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(slurmConfig)
        });
        const result = await response.json();
        if (response.ok) {
            outputDiv.style.color = "var(--success-color, #029702)";
            let msg = `✅ ${result.message || 'Job submitted!'}\n\n`;
            msg += `   Job ID: ${result.job_id || 'N/A'}\n`;
            msg += `   Slurm Script: ${result.script_path || 'N/A'}\n\n`;
            msg += `   MONITOR (vLLM logs are in Slurm output/error files):\n`;
            msg += `   Output: tail -f ${result.slurm_output_file_pattern || 'N/A'}\n`;
            msg += `   Error:  tail -f ${result.slurm_error_file_pattern || 'N/A'}\n\n`;
            msg += `ℹ️ ${result.monitoring_note || ''}`;
            outputDiv.textContent = msg;
        } else {
            outputDiv.style.color = "var(--error-color, #ff3b30)";
            outputDiv.textContent = `❌ Error (${response.status}): ${result.detail || response.statusText || 'Unknown backend error.'}`;
        }
    } catch (error) {
        outputDiv.style.color = "var(--error-color, #ff3b30)";
        outputDiv.textContent = `❌ Network/Connection Error: ${error.message}. Is backend at ${BACKEND_API_URL} running?`;
    } finally {
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    populateStaticModels();
    const deployBtn = document.getElementById('deploy-button');
    if (deployBtn) deployBtn.addEventListener('click', handleDeployClick);
    const refreshBtn = document.getElementById('refresh-endpoints-button');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshDeployedEndpoints);
});
EOF_JS
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write JavaScript file."; exit 1; fi;
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$JS_TARGET_PATH";
echo "✅ JavaScript logic file ($JS_FILENAME) configured."
echo "";

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
        }
        body { font-family: var(--font-body); margin: 0; padding:0; background-color: var(--bg-color); color: var(--text-color); display: flex; flex-direction: column; min-height: 100vh; font-size: 16px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); color: white; padding: 20px 30px; text-align: center; border-bottom: 3px solid var(--accent-color); text-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .header h1 { margin: 0; font-family: var(--font-heading); font-size: 2.3em; font-weight: 700; letter-spacing: 2px; display: flex; align-items: center; justify-content: center; gap: 15px; }
        .header p { margin: 8px 0 0; font-size: 0.95em; opacity: 0.9; font-weight: 300;}
        .main-container { display: flex; flex-wrap: wrap; padding: 20px; gap: 20px; flex-grow: 1; max-width: 1300px; margin: 20px auto; width: 100%; box-sizing: border-box;}
        .form-container, .endpoints-container { background-color: var(--card-bg-color); padding: 25px; border-radius: 10px; border: 1px solid var(--border-color); box-sizing: border-box; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); }
        .form-container { flex: 2; min-width: 400px; }
        .endpoints-container { flex: 1; min-width: 350px; }
        h3 { font-family: var(--font-heading); color: var(--secondary-color); border-bottom: 1px solid var(--accent-color); padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; font-size: 1.4em; letter-spacing: 1px; display: flex; align-items: center; gap: 10px; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; font-weight: 600; font-size: 0.9em; color: var(--text-muted-color); text-transform: uppercase; letter-spacing: 0.5px;}
        select, input[type="text"], input[type="number"], input[type="email"], input[type="password"] { width: 100%; padding: 12px; margin-bottom: 15px; border-radius: 5px; border: 1px solid var(--border-color); box-sizing: border-box; font-size: 0.95em; background-color: #3a3a3c; color: var(--text-color); transition: border-color 0.2s ease, box-shadow 0.2s ease; }
        select:focus, input:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.3); outline: none; background-color: #4a4a4e; }
        button { background: linear-gradient(to right, var(--primary-color), var(--secondary-color)); color: white; padding: 12px 20px; cursor: pointer; border: none; border-radius: 5px; font-weight: bold; font-size: 1em; text-transform: uppercase; letter-spacing: 0.8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s ease; margin-top: 10px; width: 100%; }
        button:hover:not(:disabled), button:focus:not(:disabled) { background: linear-gradient(to right, var(--secondary-color), var(--primary-color)); box-shadow: 0 4px 10px rgba(0,0,0,0.3); transform: translateY(-2px); }
        button:disabled { background: #555; cursor: not-allowed; opacity: 0.7; }
        #output { margin-top: 20px; padding: 15px; background-color: #1c1c1e; border: 1px solid var(--border-color); border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; max-height: 400px; overflow-y: auto; line-height: 1.6; font-family: 'Courier New', Courier, monospace; color: var(--text-color); }
        .slurm-options h3 { margin-top: 25px; font-size: 1.2em;}
        .slurm-options label { font-weight: 400; font-size: 0.85em; margin-top: 8px; }
        .endpoints-container #refresh-endpoints-button { background: linear-gradient(to right, #ffcc00, #ff9500); margin-bottom: 15px; }
        .endpoints-container #refresh-endpoints-button:hover { background: linear-gradient(to right, #ff9500, #ffcc00); }
        .footer { text-align: center; padding: 20px; background-color: #0e0e0f; color: #8e8e93; font-size: 0.9em; margin-top: auto; border-top: 3px solid var(--accent-color); }
        .icon { margin-right: 8px; font-size: 1.2em; vertical-align: middle;}
    </style>
</head>
<body>
    <div class="header"><h1><span class="icon"></span> SERAPHIM <span class="icon"></span></h1><p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p></div>
    <div class="main-container">
        <div class="form-container">
            <h3><span class="icon">⚙️</span> Deploy New vLLM Instance via Backend</h3>
            <label for="model-select">Select Model:</label><select id="model-select"><option value="">-- Select a Model --</option></select>
            <label for="service-port">Service Port (on Slurm node):</label><input type="number" id="service-port" value="8000" min="1024" max="65535"/>
            <label for="hf-token">Hugging Face Token (Optional):</label><input type="password" id="hf-token" placeholder="Needed for Llama, gated models, etc."/>
            <div class="slurm-options">
                <h3>Slurm Configuration</h3>
                <label for="job-name">Job Name:</label><input type="text" id="job-name" value="vllm_service"/>
                <label for="time-limit">Time Limit (HH:MM:SS):</label><input type="text" id="time-limit" value="23:59:59"/>
                <label for="gpus">GPUs (e.g., 1 or a100:1):</label><input type="text" id="gpus" value="1"/>
                <label for="cpus-per-task">CPUs per Task:</label><input type="number" id="cpus-per-task" value="4" min="1"/>
                <label for="mem">Memory (e.g., 32G):</label><input type="text" id="mem" value="32G"/>
                <label for="mail-user">Email Notify (Optional):</label><input type="email" id="mail-user" placeholder="your_email@example.com"/>
            </div>
            <button id="deploy-button">Deploy to Slurm via Backend</button>
            <div id="output">Configure and click deploy. Status will appear here.</div>
        </div>
        <div class="endpoints-container">
            <h3><span class="icon">📡</span> Active Deployments (Placeholder)</h3>
            <button id="refresh-endpoints-button">Refresh Status</button>
            <div id="deployed-endpoints-list"><p>Functionality to list active Slurm jobs requires further backend development.</p></div>
        </div>
    </div>
    <div class="footer">✧ SERAPHIM CORE Interface v1.3 (Foreground Slurm Service) ✧ System Online ✧</div>
    <script src="seraphim_logic.js" defer></script>
</body>
</html>
EOF_HTML
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write HTML file."; exit 1; fi;
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

BACKEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_backend.log"
FRONTEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.log"
BACKEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_backend.pid"
FRONTEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.pid"

echo "Starting SERAPHIM Application..."
if [ -f "\$BACKEND_PID_FILE" ] && ps -p \$(cat "\$BACKEND_PID_FILE") > /dev/null; then
    echo "❌ Backend already running (PID: \$(cat "\$BACKEND_PID_FILE")). Use ./stop_seraphim.sh first."
    exit 1
fi
cd "\$SERAPHIM_DIR_START" || { echo "Error: Could not navigate to \$SERAPHIM_DIR_START"; exit 1; }
echo "Activating Conda: \$CONDA_ENV_NAME_START..."
_CONDA_SH_PATH="\$CONDA_BASE_PATH_START/etc/profile.d/conda.sh"
if [ -z "\$CONDA_BASE_PATH_START" ]; then
    _FALLBACK_CONDA_BASE_PATH=\$(conda info --base 2>/dev/null)
    if [ -n "\$_FALLBACK_CONDA_BASE_PATH" ]; then _CONDA_SH_PATH="\$_FALLBACK_CONDA_BASE_PATH/etc/profile.d/conda.sh"; fi
fi
if [ ! -f "\$_CONDA_SH_PATH" ]; then echo "Error: conda.sh not found. Cannot activate."; exit 1; fi
# shellcheck source=/dev/null
. "\$_CONDA_SH_PATH"; conda activate "\$CONDA_ENV_NAME_START"
if [ "\$CONDA_DEFAULT_ENV" != "\$CONDA_ENV_NAME_START" ]; then echo "Error: Failed to activate conda env."; exit 1; fi
echo "Conda env '\$CONDA_ENV_NAME_START' activated."

echo "Starting Backend Server (port \$BACKEND_PORT_START)... Log: \$BACKEND_LOG_FILE"
nohup python "\$BACKEND_SCRIPT_START" > "\$BACKEND_LOG_FILE" 2>&1 &
_BACKEND_PID=\$!; echo \$_BACKEND_PID > "\$BACKEND_PID_FILE"
echo "Backend PID: \$_BACKEND_PID."
sleep 3; if ! ps -p \$_BACKEND_PID > /dev/null; then echo "❌ Error: Backend failed to start."; rm -f "\$BACKEND_PID_FILE"; exit 1; fi

echo "Starting Frontend Server (port \$FRONTEND_PORT_START)... Log: \$FRONTEND_LOG_FILE"
nohup python -m http.server --bind 0.0.0.0 "\$FRONTEND_PORT_START" > "\$FRONTEND_LOG_FILE" 2>&1 &
_FRONTEND_PID=\$!; echo \$_FRONTEND_PID > "\$FRONTEND_PID_FILE"
echo "Frontend PID: \$_FRONTEND_PID."
sleep 1; if ! ps -p \$_FRONTEND_PID > /dev/null; then echo "❌ Error: Frontend failed to start."; kill \$_BACKEND_PID; rm -f "\$BACKEND_PID_FILE" "\$FRONTEND_PID_FILE"; exit 1; fi

_SERVER_IP=\$(hostname -I | awk '{print \$1}' || echo "YOUR_SERVER_IP")
echo "================================="
echo "✅ SERAPHIM Application Started!"
echo "Access Frontend: http://\${_SERVER_IP}:\$FRONTEND_PORT_START"
echo "To stop: ./$STOP_SCRIPT_FILENAME"
echo "================================="
EOF_START_SCRIPT
chmod +x "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_BASE_PATH_PLACEHOLDER}}|$ESCAPED_CONDA_BASE_PATH_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{BACKEND_FILENAME_PLACEHOLDER}}|$BACKEND_FILENAME|g" "$START_SCRIPT_TARGET_PATH"
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
        if ps -p "\$_PID_TO_KILL" > /dev/null; then
            echo "Stopping \$process_name (PID: \$_PID_TO_KILL)..."; kill "\$_PID_TO_KILL"; sleep 1
            if ps -p "\$_PID_TO_KILL" > /dev/null; then kill -9 "\$_PID_TO_KILL"; sleep 1; fi
            if ps -p "\$_PID_TO_KILL" > /dev/null; then echo "❌ Error stopping \$process_name."; else echo "✅ \$process_name stopped."; fi
        else echo "ℹ️ \$process_name (PID \$_PID_TO_KILL) not running."; fi
        rm -f "\$pid_file"
    else echo "⚠️ \$process_name PID file not found."; fi
}
stop_process "\$BACKEND_PID_FILE_STOP" "Backend Server"
stop_process "\$FRONTEND_PID_FILE_STOP" "Frontend Server"
echo "SERAPHIM Stop Attempted."
EOF_STOP_SCRIPT
chmod +x "$STOP_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$STOP_SCRIPT_TARGET_PATH"
echo "✅ Stop script ($STOP_SCRIPT_FILENAME) created."
echo ""

echo "======================================================================"
echo "✅ SERAPHIM Setup Complete!"
echo "To run: cd \"$SERAPHIM_DIR\" && ./$START_SCRIPT_FILENAME"
echo "To stop: cd \"$SERAPHIM_DIR\" && ./$STOP_SCRIPT_FILENAME"
_SERVER_IP_FINAL=\$(hostname -I | awk '{print \$1}' || echo "YOUR_SERVER_IP")
echo "Access UI: http://\${_SERVER_IP_FINAL}:$FRONTEND_PORT"
echo "======================================================================"
echo "🚨 Notes: User running backend needs sbatch access. Review CORS in $BACKEND_FILENAME for production."
echo "======================================================================"
exit 0
