#!/bin/bash

# SERAPHIM Installation Script

# Variables
CONDA_ENV_NAME="seraphim_vllm_env"
SERAPHIM_DIR="$HOME/SERAPHIM"
FRONTEND_DIR="$SERAPHIM_DIR/frontend"
SCRIPTS_DIR="$SERAPHIM_DIR/scripts"
VLLM_REQUIREMENTS_FILE="vllm_requirements.txt"

echo "Starting SERAPHIM vLLM Deployment Setup..."
echo "=========================================="

# --- Helper Functions ---
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed. Please install $1 and re-run the script."
        exit 1
    fi
}

# --- Pre-flight Checks ---
echo "Checking for conda..."
check_command conda

# --- Create SERAPHIM Directories ---
echo "Creating SERAPHIM directories at $SERAPHIM_DIR..."
mkdir -p "$FRONTEND_DIR"
mkdir -p "$SCRIPTS_DIR"
echo "Directories created."
echo ""

# --- Create vLLM Requirements File ---
echo "Creating vLLM requirements file ($VLLM_REQUIREMENTS_FILE)..."
cat > "$SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE" << EOF
aiohappyeyeballs==2.4.4
aiohttp==3.11.11
aiohttp-cors==0.7.0
aiosignal==1.3.2
airportsdata==20241001
annotated-types==0.7.0
anyio==4.8.0
astor==0.8.1
async-timeout==5.0.1
attrs==25.1.0
blake3==1.0.4
cachetools==5.5.1
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
cloudpickle==3.1.1
colorful==0.5.6
compressed-tensors==0.9.0
depyf==0.18.0
dill==0.3.9
diskcache==5.6.3
distlib==0.3.9
distro==1.9.0
einops==0.8.0
exceptiongroup==1.2.2
fastapi==0.115.8
filelock==3.17.0
frozenlist==1.5.0
fsspec==2025.2.0
gguf==0.10.0
google-api-core==2.24.1
google-auth==2.38.0
googleapis-common-protos==1.66.0
grpcio==1.70.0
h11==0.14.0
httpcore==1.0.7
httptools==0.6.4
httpx==0.28.1
huggingface-hub==0.28.1
idna==3.10
importlib_metadata==8.6.1
iniconfig==2.0.0
interegular==0.3.3
Jinja2==3.1.5
jiter==0.8.2
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
lark==1.2.2
lm-format-enforcer==0.10.9
MarkupSafe==3.0.2
mistral_common==1.5.2
mpmath==1.3.0
msgpack==1.1.0
msgspec==0.19.0
multidict==6.1.0
nest-asyncio==1.6.0
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-ml-py==12.570.86
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
openai==1.61.0
opencensus==0.11.4
opencensus-context==0.1.3
opencv-python-headless==4.11.0.86
outlines==0.1.11
outlines_core==0.1.26
packaging==24.2
partial-json-parser==0.2.1.1.post5
pillow==10.4.0
platformdirs==4.3.6
pluggy==1.5.0
prometheus-fastapi-instrumentator==7.0.2
prometheus_client==0.21.1
propcache==0.2.1
proto-plus==1.26.0
protobuf==5.29.3
psutil==6.1.1
py-cpuinfo==9.0.0
py-spy==0.4.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
pybind11==2.13.6
pycountry==24.6.1
pydantic==2.10.6
pydantic_core==2.27.2
pytest==8.3.4
python-dotenv==1.0.1
PyYAML==6.0.2
pyzmq==26.2.1
ray==2.42.0
referencing==0.36.2
regex==2024.11.6
requests==2.32.3
rpds-py==0.22.3
rsa==4.9
safetensors==0.5.2
sentencepiece==0.2.0
six==1.17.0
smart-open==7.1.0
sniffio==1.3.1
starlette==0.45.3
sympy==1.13.1
tiktoken==0.7.0
tokenizers==0.21.0
tomli==2.2.1
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
tqdm==4.67.1
transformers==4.48.2
triton==3.1.0
typing_extensions==4.12.2
urllib3==2.3.0
uvicorn==0.34.0
uvloop==0.21.0
virtualenv==20.29.1
vllm==0.7.1
watchfiles==1.0.4
websockets==14.2
wrapt==1.17.2
xformers==0.0.28.post3
xgrammar==0.1.11
yarl==1.18.3
zipp==3.21.0
# Additional crucial dependency for GPU acceleration if not covered by torch/vllm's own nvidia packages
flash-attn
EOF
echo "Requirements file created."
echo ""

# --- Setup Conda Environment ---
echo "Setting up Conda environment: $CONDA_ENV_NAME"
conda env remove -n "$CONDA_ENV_NAME" -y || echo "Info: No pre-existing '$CONDA_ENV_NAME' env to remove or removal failed (which is ok if it didn't exist)."
conda create -n "$CONDA_ENV_NAME" python=3.10 -y

echo "Activating Conda environment: $CONDA_ENV_NAME"
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh"
    echo "Please ensure your Conda installation is correct and accessible."
    exit 1
fi
conda activate "$CONDA_ENV_NAME"

if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "Error: Failed to activate conda environment $CONDA_ENV_NAME."
    echo "Current active env: $CONDA_DEFAULT_ENV"
    echo "Please ensure conda is initialized correctly for your shell (e.g., run 'conda init bash' and restart your shell)."
    exit 1
fi
echo "Conda environment '$CONDA_ENV_NAME' activated. Python version: $(python --version)"
echo ""

# --- Install Python Dependencies ---
echo "Installing Python dependencies from $VLLM_REQUIREMENTS_FILE..."
echo "Upgrading pip, setuptools, and wheel using 'python -m pip'..."
python -m pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "Error: Failed to upgrade pip/setuptools/wheel in $CONDA_ENV_NAME."
    exit 1
fi

echo "Pre-installing PyTorch (torch, torchaudio, torchvision) as per requirements file..."
python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
if [ $? -ne 0 ]; then
    echo "Error: Failed to pre-install PyTorch packages (torch, torchaudio, torchvision) in $CONDA_ENV_NAME."
    echo "Please check your network connection and CUDA compatibility if GPU versions are intended."
    exit 1
fi
echo "PyTorch, torchaudio, and torchvision packages installed."

echo "Installing all remaining requirements from $VLLM_REQUIREMENTS_FILE (including flash-attn)..."
python -m pip install -r "$SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements from $VLLM_REQUIREMENTS_FILE in $CONDA_ENV_NAME."
    echo "If 'flash-attn' or other packages requiring compilation fail, ensure you have the necessary build tools:"
    echo "  - CUDA Toolkit (nvcc compiler, matching your drivers, e.g., CUDA 12.x)"
    echo "  - C/C++ compiler (gcc/g++)"
    echo "  - Python development headers (e.g., python3-dev or python3.10-dev)"
    exit 1
fi
echo "All Python dependencies installed."
echo ""


# --- Place Frontend HTML ---
echo "Placing frontend HTML file (seraphim_deploy.html) in $FRONTEND_DIR..."
cat > "$FRONTEND_DIR/seraphim_deploy.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SERAPHIM - vLLM Deployment</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }
        h1 { color: #0056b3; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; font-weight: bold; }
        select, input[type="text"], input[type="number"], input[type="email"], button { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border-radius: 4px; border: 1px solid #ccc; box-sizing: border-box; }
        button { background-color: #007bff; color: white; cursor: pointer; border: none; }
        button:hover { background-color: #0056b3; }
        #output { margin-top: 20px; padding: 10px; background-color: #e9ecef; border: 1px solid #ced4da; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }
        .slurm-options label { font-weight: normal; margin-top: 10px; }
        .info-text { font-size: 0.9em; color: #555; margin-bottom: 10px;}
    </style>
</head>
<body>
    <div class="container">
        <h1>SERAPHIM - vLLM Model Deployment on Slurm</h1>
        <p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p>

        <label for="vllm-api-url">vLLM API Base URL (for fetching models list):</label>
        <input type="text" id="vllm-api-url" value="http://localhost:8000">
        <p class="info-text">Enter the URL from which your browser can reach the vLLM <code>/v1/models</code> endpoint. This might be <code>http://SERVER_IP:PORT</code> if vLLM runs on the same server as this interface, or requires an SSH tunnel (e.g., <code>http://localhost:TUNNEL_PORT</code>) if vLLM runs on a compute node.</p>
        <button id="fetch-models-button">Fetch/Refresh Models</button>

        <label for="model-select">Select Model:</label>
        <select id="model-select">
            <option value="">Click "Fetch/Refresh Models" or enter API URL first...</option>
        </select>

        <label for="service-port">Service Port for new vLLM (on Slurm compute node):</label>
        <input type="number" id="service-port" value="8092">

        <label for="hf-token">Hugging Face Token (HF_TOKEN):</label>
        <input type="text" id="hf-token" placeholder="Enter your Hugging Face Token">

        <h3>Slurm Configuration (Optional Overrides)</h3>
        <div class="slurm-options">
            <label for="job-name">Job Name:</label>
            <input type="text" id="job-name" value="vllm_service">
            <label for="time-limit">Time Limit (hh:mm:ss):</label>
            <input type="text" id="time-limit" value="23:59:59">
            <label for="gpus">GPUs Requested:</label>
            <input type="number" id="gpus" value="1">
            <label for="cpus-per-task">CPUs per Task:</label>
            <input type="number" id="cpus-per-task" value="4">
            <label for="mem">Memory per Node (e.g., 16G):</label>
            <input type="text" id="mem" value="16G">
            <label for="mail-user">Email for Notifications (Optional):</label>
            <input type="email" id="mail-user" placeholder="your_email@example.com">
        </div>

        <button id="deploy-button">Generate Deploy Command</button>
        <div id="output">Click "Generate Deploy Command" to see the sbatch script content.</div>
    </div>

    <script>
        const modelSelect = document.getElementById('model-select');
        const deployButton = document.getElementById('deploy-button');
        const outputDiv = document.getElementById('output');
        const servicePortInput = document.getElementById('service-port');
        const hfTokenInput = document.getElementById('hf-token');
        const vllmApiUrlInput = document.getElementById('vllm-api-url');
        const fetchModelsButton = document.getElementById('fetch-models-button');

        const jobNameInput = document.getElementById('job-name');
        const timeLimitInput = document.getElementById('time-limit');
        const gpusInput = document.getElementById('gpus');
        const cpusPerTaskInput = document.getElementById('cpus-per-task');
        const memInput = document.getElementById('mem');
        const mailUserInput = document.getElementById('mail-user');
        
        const CONDA_ENV_NAME_JS = "{{CONDA_ENV_NAME_PLACEHOLDER}}";
        const SCRIPTS_DIR_JS = "{{SCRIPTS_DIR_PLACEHOLDER}}";

        async function fetchModels() {
            const baseUrl = vllmApiUrlInput.value.trim();
            if (!baseUrl) {
                modelSelect.innerHTML = '<option value="">Please enter a vLLM API Base URL.</option>';
                return;
            }
            const fullApiUrl = baseUrl.endsWith('/') ? baseUrl + 'v1/models' : baseUrl + '/v1/models';
            
            modelSelect.innerHTML = '<option value="">Fetching models from ' + fullApiUrl + '...</option>';
            try {
                const response = await fetch(fullApiUrl);
                if (!response.ok) {
                    let errorMsg = response.statusText;
                    try { const errorData = await response.json(); errorMsg = errorData.detail || errorData.message || errorMsg; } catch (e) {}
                    throw new Error(`HTTP error ${response.status}: ${errorMsg}`);
                }
                const data = await response.json();
                
                modelSelect.innerHTML = '<option value="">-- Select a Model --</option>';
                if (data.data && data.data.length > 0) {
                    data.data.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id; option.textContent = model.id;
                        modelSelect.appendChild(option);
                    });
                } else {
                     modelSelect.innerHTML = '<option value="">No models found. API response format might be unexpected. Check console.</option>';
                     console.warn("Received data from " + fullApiUrl + ":", data, "Expected format: { data: [{id: 'model_name'}, ...] }");
                }
            } catch (error) {
                console.error("Error fetching models from " + fullApiUrl + ":", error);
                modelSelect.innerHTML = `<option value="">Error fetching models. Is vLLM server at ${fullApiUrl} reachable? Details: ${error.message}</option>`;
                const fallbackModels = ["mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Pixtral-12B-2409", "mistral-community/pixtral-12b", "google/gemma-7b-it"];
                modelSelect.innerHTML += fallbackModels.map(modelId => `<option value="${modelId}">${modelId} (Fallback)</option>`).join('');
            }
        }

        fetchModelsButton.addEventListener('click', fetchModels);

        deployButton.addEventListener('click', () => {
            const selectedModel = modelSelect.value;
            const servicePort = servicePortInput.value;
            const hfToken = hfTokenInput.value;
            const jobName = jobNameInput.value;
            const timeLimit = timeLimitInput.value;
            const gpus = gpusInput.value;
            const cpusPerTask = cpusPerTaskInput.value;
            const mem = memInput.value;
            const mailUser = mailUserInput.value;

            if (!selectedModel) { outputDiv.textContent = "Please select a model first."; return; }
            if (!servicePort) { outputDiv.textContent = "Please enter a service port for the new vLLM instance."; return; }
            if (!hfToken && selectedModel && (selectedModel.toLowerCase().includes("pixtral") || selectedModel.toLowerCase().includes("gated_model_identifier_example"))) {
                outputDiv.textContent = "Please enter your Hugging Face Token for gated models."; return;
            }

            let vllmServeCommand = `vllm serve "${selectedModel}"`; 
            let modelArgs = [`--host "0.0.0.0" --port ${servicePort}`, `--disable-log-stats`, `--trust-remote-code`];
            let maxModelLen = 16384; 

            if (selectedModel.toLowerCase().includes("pixtral")) {
                modelArgs.push(`--guided-decoding-backend=lm-format-enforcer`, `--limit_mm_per_prompt 'image=8'`);
                if (selectedModel.includes("mistralai/Pixtral-12B-2409")) {
                     modelArgs.push(`--enable-auto-tool-choice`, `--tool-call-parser=mistral`, `--tokenizer_mode mistral`, `--revision aaef4baf771761a81ba89465a18e4427f3a105f9`);
                } else if (selectedModel.includes("mistral-community/pixtral-12b")) {
                     modelArgs.push(`--revision c2756cbbb9422eba9f6c5c439a214b0392dfc998`); maxModelLen = 32768; 
                }
            }
            modelArgs.push(`--max-model-len ${maxModelLen}`);
            vllmServeCommand += " \\\n    " + modelArgs.join(" \\\n    ");

            const mailTypeLine = mailUser ? `#SBATCH --mail-type=ALL\\n#SBATCH --mail-user=\${mailUser}` : "#SBATCH --mail-type=NONE";
            const condaBasePathForSlurm = "\\$(conda info --base)"; 
            const seraphimScriptsDirForSlurm = SCRIPTS_DIR_JS;

            const sbatchScriptContent = \`#!/bin/bash
#SBATCH --job-name=\${jobName}
#SBATCH --output=\${seraphimScriptsDirForSlurm}/\${jobName}_%j.out
#SBATCH --error=\${seraphimScriptsDirForSlurm}/\${jobName}_%j.err
#SBATCH --time=\${timeLimit}
#SBATCH --gres=gpu:\${gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=\${cpusPerTask}
#SBATCH --mem=\${mem}
\${mailTypeLine}

echo "=================================================================="
echo "SERAPHIM vLLM Deployment Job"
echo "Job Name: \${SLURM_JOB_NAME}; Job ID: \${SLURM_JOB_ID}"
echo "Submitted to partition: \${SLURM_JOB_PARTITION}; Running on host: $(hostname)"
echo "Working directory: $(pwd); Script/Log directory: \${seraphimScriptsDirForSlurm}"
echo "Deployed Model: ${selectedModel}; Service Port: ${servicePort}"
echo "Timestamp: $(date); Conda Env Name: \${CONDA_ENV_NAME_JS}"
echo "=================================================================="

CONDA_BASE_PATH_SLURM="\${condaBasePathForSlurm}"
CONDA_SH_PATH="\${CONDA_BASE_PATH_SLURM}/etc/profile.d/conda.sh"

echo "Attempting to source Conda from: \${CONDA_SH_PATH}"
if [ -f "\${CONDA_SH_PATH}" ]; then source "\${CONDA_SH_PATH}"; echo "Sourced conda.sh from \${CONDA_BASE_PATH_SLURM}";
else
    echo "WARN: conda.sh not found at \${CONDA_SH_PATH}. Trying common alternatives..."
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh"; echo "Sourced $HOME/miniconda3/etc/profile.d/conda.sh";
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then source "$HOME/anaconda3/etc/profile.d/conda.sh"; echo "Sourced $HOME/anaconda3/etc/profile.d/conda.sh";
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then source "/opt/conda/etc/profile.d/conda.sh"; echo "Sourced /opt/conda/etc/profile.d/conda.sh";
    else echo "ERROR: conda.sh not found. Conda environment activation will likely fail."; fi
fi

echo "Attempting to activate Conda environment: \${CONDA_ENV_NAME_JS}"
conda activate \${CONDA_ENV_NAME_JS}
if [ "$CONDA_DEFAULT_ENV" != "\${CONDA_ENV_NAME_JS}" ]; then
    echo "ERROR: Failed to activate conda environment '\${CONDA_ENV_NAME_JS}'. Current: $CONDA_DEFAULT_ENV. Expected: \${CONDA_ENV_NAME_JS}";
else echo "Successfully activated conda environment: \${CONDA_ENV_NAME_JS}. Python: $(which python) ($(python --version))"; fi

export MAX_JOBS=16 HF_TOKEN="\${hfToken}" VLLM_USE_TRITON_FLASH_ATTN="True" VLLM_CONFIGURE_LOGGING=0 VLLM_NO_USAGE_STATS="True" VLLM_DO_NOT_TRACK="True" VLLM_USE_V1=1 
echo -e "Starting vLLM server with command:\\n\${vllmServeCommand}"
\${vllmServeCommand}
echo "=================================================================="
echo "vLLM Server command launched. Check Slurm logs in \${seraphimScriptsDirForSlurm}"
echo "Job finished script execution at $(date)"
echo "=================================================================="
\`;
            const deployScriptName = "deploy_vllm_job.slurm";
            outputDiv.textContent = \`To deploy, save the following content as \${SCRIPTS_DIR_JS}/\${deployScriptName}:\n\n-------------------------------------\n\${sbatchScriptContent}\n-------------------------------------\n\nThen make it executable and submit:\n\nchmod +x \${SCRIPTS_DIR_JS}/\${deployScriptName}\nsbatch \${SCRIPTS_DIR_JS}/\${deployScriptName}\n\nOutput and error files will be in \${SCRIPTS_DIR_JS}\`;
        });
        // document.addEventListener('DOMContentLoaded', fetchModels); // Fetch on load removed, now button triggered
    </script>
</body>
</html>
EOF

ESCAPED_CONDA_ENV_NAME=$(printf '%s\n' "$CONDA_ENV_NAME" | sed 's:[&/\]:\\&:g')
ESCAPED_SCRIPTS_DIR=$(printf '%s\n' "$SCRIPTS_DIR" | sed 's:[&/\]:\\&:g')
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME|g" "$FRONTEND_DIR/seraphim_deploy.html"
sed -i "s|{{SCRIPTS_DIR_PLACEHOLDER}}|$ESCAPED_SCRIPTS_DIR|g" "$FRONTEND_DIR/seraphim_deploy.html"

echo "Frontend HTML (seraphim_deploy.html) created and configured in $FRONTEND_DIR."
echo ""

# --- Final Instructions ---
echo "=========================================="
echo "SERAPHIM Setup Complete!"
echo "=========================================="
echo ""
echo "To use SERAPHIM:"
echo ""
echo "1. Activate the Conda environment (if not already active for your current shell):"
echo "   source $CONDA_BASE/etc/profile.d/conda.sh" 
echo "   conda activate $CONDA_ENV_NAME"
echo ""
echo "2. Start the HTTP server ON THE SERVER ($HOSTNAME) to serve the frontend HTML:"
echo "   cd \"$FRONTEND_DIR\""
echo "   python -m http.server --bind 0.0.0.0 8869  (Or your server's specific IP: --bind YOUR_SERVER_IP 8869)"
echo "   Then, from YOUR LOCAL MACHINE, open http://YOUR_SERVER_IP:8869 (e.g., http://10.16.256.2:8869) in your web browser."
echo "   Ensure port 8869 (or your chosen port) is open in the server's firewall if necessary."
echo ""
echo "3. Configure vLLM API URL in the HTML Interface:"
echo "   - In the browser, enter the correct 'vLLM API Base URL' from which your browser can fetch the model list."
echo "   - Example Scenarios for this URL:"
echo "     a) If you run a separate vLLM instance on $HOSTNAME (port 8000) just for model listing: http://$HOSTNAME:8000"
echo "     b) If the vLLM service is deployed by Slurm on a COMPUTE_NODE (port 8092), you'll need a way to reach it from your browser:"
echo "        - Direct Access (if network allows): http://COMPUTE_NODE_IP:8092"
echo "        - SSH Tunnel (from your local machine):"
echo "          ssh -L 8000:COMPUTE_NODE_IP:8092 your_user@YOUR_SERVER_IP"
echo "          Then use: http://localhost:8000 in the HTML field (localhost refers to your machine)."
echo "   - Click 'Fetch/Refresh Models' after setting the URL."
echo ""
echo "4. The 'Generate Deploy Command' button creates Slurm batch script content."
echo "   - Copy this to a file ON THE SERVER (e.g., $SCRIPTS_DIR/deploy_vllm_job.slurm)."
echo "   - Make executable: chmod +x $SCRIPTS_DIR/deploy_vllm_job.slurm"
echo "   - Submit: sbatch $SCRIPTS_DIR/deploy_vllm_job.slurm"
echo ""
echo "5. SLURM SCRIPT NOTES:"
echo "   - Output/error files will be in '$SCRIPTS_DIR'."
echo "   - Ensure Slurm nodes can access Conda environment: $CONDA_BASE/envs/$CONDA_ENV_NAME"
echo ""
echo "Location of SERAPHIM files: $SERAPHIM_DIR"
echo "Frontend: $FRONTEND_DIR/seraphim_deploy.html"
echo "Scripts & Logs: $SCRIPTS_DIR"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "=========================================="

if [ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]; then
    conda deactivate
    echo "Deactivated $CONDA_ENV_NAME for current script session."
fi