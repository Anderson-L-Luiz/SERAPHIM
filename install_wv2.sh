#!/bin/bash

# SERAPHIM Installation Script - Incorporating User's JS Fix & Digital Divine Theme

# Exit immediately if a command exits with a non-zero status.
set -e 

# Variables
CONDA_ENV_NAME="seraphim_vllm_env";
SERAPHIM_DIR="$HOME/SERAPHIM"; # Main directory for install.sh, scripts, and html
SCRIPTS_DIR="$SERAPHIM_DIR/scripts"; # For Slurm scripts & logs
HTML_FILENAME="seraphim_deploy.html";
JS_FILENAME="seraphim_logic.js";
HTML_TARGET_PATH="$SERAPHIM_DIR/$HTML_FILENAME";
JS_TARGET_PATH="$SERAPHIM_DIR/$JS_FILENAME"; # JS file will be alongside HTML
VLLM_REQUIREMENTS_FILE="vllm_requirements.txt";

# --- Initial Check ---
if ! command -v conda &> /dev/null; then
     echo "❌ Error: conda command not found. Please ensure Conda (Miniconda/Anaconda) is installed and in your PATH."; exit 1;
fi;
# Ensure running from base
# CURRENT_ENV=$(conda info | grep "active environment" | awk '{print $4}' || true); 
# Deactivating this check as it caused issues before. User should ensure they run from base.

echo "Starting SERAPHIM vLLM Deployment Setup...";
echo "Target Directory: $SERAPHIM_DIR";
echo "==========================================";

# --- Create Directories ---
echo "Ensuring directories exist...";
mkdir -p "$SERAPHIM_DIR";
mkdir -p "$SCRIPTS_DIR";
echo "Directories checked/created.";
echo "";

# --- Create vLLM Requirements File ---
echo "Creating requirements file: $SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE";
# Using user's minimal list + flash-attn + core vllm dependencies from previous versions
cat > "$SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE" << EOF
# Core vLLM and serving
vllm==0.7.1
uvicorn==0.34.0
fastapi==0.115.8
# Dependencies likely needed by vLLM or models (based on previous full list)
aiohappyeyeballs==2.4.4
aiohttp==3.11.11
aiohttp-cors==0.7.0
huggingface-hub==0.28.1
numpy==1.26.4
openai==1.61.0
packaging==24.2
prometheus-client==0.21.1
protobuf==5.29.3
pydantic==2.10.6 # vLLM 0.7.1 might need pydantic v1 or handle v2 carefully
pydantic_core==2.27.2 
python-dotenv==1.0.1
PyYAML==6.0.2
ray==2.42.0 # Check vLLM docs for compatible Ray version if using distributed
requests==2.32.3
safetensors==0.5.2
sentencepiece==0.2.0
tokenizers==0.21.0
torch==2.5.1 # Handled separately below
torchaudio==2.5.1 # Handled separately below
torchvision==0.20.1 # Handled separately below
transformers==4.48.2
typing_extensions==4.12.2
# Optional GPU acceleration (Install if hardware supports & compilation works)
# flash-attn 
xformers==0.0.28.post3 # Often beneficial with NVIDIA GPUs
EOF
echo "Requirements file created.";
echo "";


# --- Setup Conda Environment ---
echo "Setting up Conda environment: $CONDA_ENV_NAME";
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Removing existing Conda environment: $CONDA_ENV_NAME";
    conda env remove -n "$CONDA_ENV_NAME" -y;
fi;
echo "Creating new Conda environment: $CONDA_ENV_NAME with Python 3.10";
conda create -n "$CONDA_ENV_NAME" python=3.10 -y;

echo "Sourcing conda for activation...";
CONDA_BASE_PATH=$(conda info --base);
CONDA_SH_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh";
if [ ! -f "$CONDA_SH_PATH" ]; then
     echo "❌ Error: conda.sh not found at $CONDA_SH_PATH"; exit 1;
fi;
. "$CONDA_SH_PATH"; # Source conda

echo "Activating Conda environment: $CONDA_ENV_NAME (for script)";
conda activate "$CONDA_ENV_NAME";
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "❌ Error: Failed to activate conda environment '$CONDA_ENV_NAME'."; exit 1;
fi;
echo "✅ Conda environment '$CONDA_ENV_NAME' activated.";
echo "";

# --- Install Python Dependencies ---
echo "Installing Python dependencies into '$CONDA_ENV_NAME'...";
python -m pip install --upgrade pip setuptools wheel;

# Install PyTorch (adjust for your CUDA version if necessary)
echo "Installing PyTorch (with CUDA 12.1 support)...";
python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121;
if [ $? -ne 0 ]; then echo "❌ Error: Failed PyTorch install. Check CUDA compatibility or network."; exit 1; fi;

echo "Installing main requirements from $SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE...";
python -m pip install -r "$SCRIPTS_DIR/$VLLM_REQUIREMENTS_FILE";
if [ $? -ne 0 ]; then echo "❌ Error: Failed main requirements install."; exit 1; fi;

# Optionally install flash-attn separately if desired and hardware/compilers are available
# echo "Attempting optional flash-attn install...";
# python -m pip install flash-attn --no-build-isolation || echo "⚠️ Warning: flash-attn install failed (optional)."

echo "✅ All Python dependencies installed.";
echo "";


# --- Generate Separate JavaScript File (Using User's Provided Code) ---
echo "Generating JavaScript logic file: $JS_TARGET_PATH";
# Use 'EOF' to prevent shell expansion within the JS code block
# PASTE THE USER'S CORRECTED JAVASCRIPT HERE
cat > "$JS_TARGET_PATH" << 'EOF'
// --- Constants ---
// These placeholders will be replaced by sed in install.sh
const CONDA_ENV_NAME_JS = "{{CONDA_ENV_NAME_PLACEHOLDER}}";
const SCRIPTS_DIR_JS = "{{SCRIPTS_DIR_PLACEHOLDER}}";

const staticModels = [
    { id: "mistralai/Mistral-7B-Instruct-v0.1", name: "Mistral-7B-Instruct-v0.1" },
    { id: "meta-llama/Llama-2-7b-chat-hf", name: "Llama-2-7B-Chat-HF" },
    { id: "google/gemma-7b-it", name: "Gemma-7B-IT (Google)" },
    { id: "Qwen/Qwen2-7B-Instruct", name: "Qwen2-7B-Instruct" },
    { id: "BAAI/AquilaChat-7B", name: "AquilaChat-7B (BAAI)"},
    { id: "mistralai/Mixtral-8x7B-Instruct-v0.1", name: "Mixtral-8x7B-Instruct-v0.1" },
    { id: "EleutherAI/gpt-j-6b", name: "GPT-J-6B (EleutherAI)"},
    { id: "tiiuae/falcon-7b-instruct", name: "Falcon-7B-Instruct (TII UAE)"},
    { id: "mistralai/Pixtral-12B-2409", name: "Pixtral-12B-2409 (Multimodal)" }, // Example, might require specific vLLM version
    { id: "microsoft/phi-2", name: "Phi-2 (Microsoft)"}
];

// --- Utility Functions ---
/**
 * Escapes characters in a string to be safely used as a shell argument or in a shell script.
 * @param {string} str The input string.
 * @returns {string} The escaped string.
 */
function shellEscape(str) {
    if (str === null || typeof str === 'undefined') {
        return '';
    }
    // Using JSON.stringify is a robust way to quote and escape for shell.
    // It will wrap the string in double quotes and handle internal quotes/escapes.
    // We remove the outer quotes added by stringify as we often add quotes in the command template itself.
    let escaped = JSON.stringify(String(str));
    return escaped.substring(1, escaped.length - 1); 
}


// --- Core Functions ---
function populateStaticModels() {
    console.log("SERAPHIM_DEBUG: Populating models...");
    const modelSelect = document.getElementById('model-select');
    if (!modelSelect) {
        console.error("SERAPHIM_DEBUG: Cannot find 'model-select' element!");
        return;
    }
    
    // Clear existing options (keep placeholder)
    while (modelSelect.options.length > 1) {
        modelSelect.remove(modelSelect.options.length - 1);
    }
    
    staticModels.forEach(model => {
        if (typeof model.id === 'string' && typeof model.name === 'string') {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        } else {
            console.warn("SERAPHIM_DEBUG: Skipping invalid model data:", model);
        }
    });
    console.log("SERAPHIM_DEBUG: Models populated. Count:", modelSelect.options.length -1); // -1 for placeholder
}

function refreshDeployedEndpoints() {
    console.log("SERAPHIM_DEBUG: Refreshing endpoints (placeholder)...");
    const listDiv = document.getElementById('deployed-endpoints-list');
    if (!listDiv) {
        console.error("SERAPHIM_DEBUG: Cannot find 'deployed-endpoints-list' div.");
        return;
    }
    listDiv.innerHTML = "<p><em>Refreshing... (This is a placeholder. Backend integration needed to list actual Slurm jobs.)</em></p>";
    // Example of how an endpoint might be displayed (static for now)
    const exampleHtml = `
        <p>Requires backend integration with Slurm (e.g., via 'squeue' or a custom API).</p>
        <div class="endpoint-item">
            <strong>Model:</strong> example/Sample-7B-Demo<br/>
            <strong>Job ID:</strong> 12345 (Example)<br/>
            <strong>Node:</strong> cnode01.example (Example)<br/>
            <strong>Port:</strong> 8000 (Example)<br/>
            <strong>Access:</strong> <a href="#" onclick="alert('This is a static example. Actual link would depend on the deployed node and port.'); return false;">http://cnode01.example:8000</a>
        </div>`;
    setTimeout(() => { listDiv.innerHTML = exampleHtml; }, 500);
}

function handleDeployClick() {
    console.log("SERAPHIM_DEBUG: Deploy clicked.");
    // Get elements fresh each time
    const modelSelect = document.getElementById('model-select');
    const outputDiv = document.getElementById('output');
    const servicePortInput = document.getElementById('service-port');
    const hfTokenInput = document.getElementById('hf-token'); // Optional
    const jobNameInput = document.getElementById('job-name');
    const timeLimitInput = document.getElementById('time-limit');
    const gpusInput = document.getElementById('gpus');
    const cpusPerTaskInput = document.getElementById('cpus-per-task');
    const memInput = document.getElementById('mem');
    const mailUserInput = document.getElementById('mail-user'); // Optional

    // Get values & apply defaults/trimming
    const selectedModel = modelSelect ? modelSelect.value : '';
    const servicePort = (servicePortInput && servicePortInput.value.trim()) ? servicePortInput.value.trim() : '8000'; // Default port
    const hfToken = hfTokenInput ? hfTokenInput.value.trim() : '';
    const jobName = (jobNameInput && jobNameInput.value.trim()) ? jobNameInput.value.trim() : 'vllm_seraph_job';
    const timeLimit = (timeLimitInput && timeLimitInput.value.trim()) ? timeLimitInput.value.trim() : '23:59:59';
    const gpus = (gpusInput && gpusInput.value.trim()) ? gpusInput.value.trim() : '1';
    const cpusPerTask = (cpusPerTaskInput && cpusPerTaskInput.value.trim()) ? cpusPerTaskInput.value.trim() : '4';
    const mem = (memInput && memInput.value.trim()) ? memInput.value.trim() : '16G';
    const mailUser = mailUserInput ? mailUserInput.value.trim() : '';

    // --- Validation ---
    if (!selectedModel) {
        outputDiv.textContent = "⚠️ Please select a model first.";
        return;
    }
    if (!/^\d+$/.test(servicePort) || parseInt(servicePort, 10) < 1024 || parseInt(servicePort, 10) > 65535) {
        outputDiv.textContent = "⚠️ Please enter a valid service port (1024-65535).";
        return;
    }
    // Basic validation for other Slurm parameters can be added here (e.g., time format, mem format)

    let warning = "";
    if (!hfToken && (selectedModel.toLowerCase().includes("llama") || selectedModel.toLowerCase().includes("meta-llama") || selectedModel.toLowerCase().includes("gated"))) {
        warning = "ℹ️ This model might be gated. Provide an HF Token if needed for access.\\n\\n";
    }
    outputDiv.textContent = warning + "Generating deployment script...";

    // --- Build vLLM Serve Command ---
    // Using the specific entrypoint mentioned by user's JS
    let vllmServeCommand = 'python -m vllm.entrypoints.openai.api_server'; 
    // Use shellEscape for values going into the command string arguments
    let modelArgs = [
        '--model "' + shellEscape(selectedModel) + '"', // Quote and escape
        '--host "0.0.0.0"',
        '--port ' + shellEscape(servicePort),
        '--disable-log-stats',
        '--trust-remote-code' 
    ];
    
    let maxModelLen = 16384; // Default
    if (selectedModel.toLowerCase().includes("mixtral")) { 
        maxModelLen = 32768; 
    }
    // Add other model-specific arguments if needed (use shellEscape if value comes from user input)
    if (selectedModel.toLowerCase().includes("pixtral")) { 
         // modelArgs.push('--guided-decoding-backend=lm-format-enforcer'); 
         // modelArgs.push('--limit_mm_per_prompt \'image=8\''); // Example
    }
    modelArgs.push('--max-model-len ' + maxModelLen); // maxModelLen is already a number here
    
    // Join arguments, ensuring proper line continuation for shell readability
    vllmServeCommand += " " + modelArgs.join(" \\\n    "); 

    // --- Build Slurm Script ---
    // Escape user-provided values that will go into the Slurm script
    const safeJobName = shellEscape(jobName);
    const safeTimeLimit = shellEscape(timeLimit);
    const safeGpus = shellEscape(gpus); // GPU spec can be complex, basic escape here
    const safeCpusPerTask = shellEscape(cpusPerTask);
    const safeMem = shellEscape(mem); // Memory spec might need more robust validation/escaping
    const safeMailUser = mailUser ? shellEscape(mailUser) : ""; 
    const safeSelectedModel = shellEscape(selectedModel); // For display
    const safeHfToken = shellEscape(hfToken); // For export

    // Construct #SBATCH lines - use concatenation for simplicity here
    const mailTypeLine = safeMailUser ? '#SBATCH --mail-type=ALL\\n#SBATCH --mail-user=' + safeMailUser : "#SBATCH --mail-type=NONE";
    // This will execute `conda info --base` inside the Slurm job
    const condaBasePathForSlurm = '$(conda info --base)'; 

    // Use template literal for the main Slurm script body for multi-line readability
    // Note: We escape shell variables meant to be evaluated *inside* the Slurm script (e.g., ${SLURM_JOB_ID}) 
    // by using a backslash \${...} IF they are inside the JS template literal.
    // Variables from the JS scope (like safeJobName, servicePort) are interpolated directly using ${...}.
    const sbatchScriptContent = `#!/bin/bash
#SBATCH --job-name=${safeJobName}
#SBATCH --output=${SCRIPTS_DIR_JS}/${safeJobName}_%j.out
#SBATCH --error=${SCRIPTS_DIR_JS}/${safeJobName}_%j.err
#SBATCH --time=${safeTimeLimit}
#SBATCH --gres=gpu:${safeGpus} 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${safeCpusPerTask}
#SBATCH --mem=${safeMem} 
${mailTypeLine}

echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job ✝"
echo "Job Start Time: $(date)"
echo "Job ID: \${SLURM_JOB_ID} running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Model: ${safeSelectedModel}"
echo "Service Port: ${servicePort}" # servicePort is safe (number)
echo "Conda Env: ${CONDA_ENV_NAME_JS}" # Placeholder replaced by sed
echo "Scripts Dir: ${SCRIPTS_DIR_JS}" # Placeholder replaced by sed
echo "=================================================================="

# Source Conda
CONDA_BASE_PATH_SLURM="${condaBasePathForSlurm}" 
if [ -z "\${CONDA_BASE_PATH_SLURM}" ]; then echo "ERROR: Could not determine Conda base path."; exit 1; fi
CONDA_SH_PATH="\${CONDA_BASE_PATH_SLURM}/etc/profile.d/conda.sh"

if [ -f "\${CONDA_SH_PATH}" ]; then
    echo "Sourcing Conda from: \${CONDA_SH_PATH}"
    . "\${CONDA_SH_PATH}" # Use . to source
else
    echo "WARN: conda.sh not found at \${CONDA_SH_PATH}."
fi

echo "Activating Conda Environment: ${CONDA_ENV_NAME_JS}"
conda activate "${CONDA_ENV_NAME_JS}" # Quoted

if [[ "\\$CONDA_PREFIX" != *"${CONDA_ENV_NAME_JS}"* ]]; then # Escaped $CONDA_PREFIX
    echo "ERROR: Failed to activate conda environment '${CONDA_ENV_NAME_JS}'. CONDA_PREFIX=\\$CONDA_PREFIX"; exit 1;
else
    echo "Conda environment '${CONDA_ENV_NAME_JS}' activated. Path: \\$CONDA_PREFIX";
fi

# Set HF Token
if [ -n "${safeHfToken}" ]; then export HF_TOKEN="${safeHfToken}"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

# Set vLLM Env Vars
export VLLM_USE_TRITON_FLASH_ATTN="True";
export VLLM_CONFIGURE_LOGGING="0";
export VLLM_NO_USAGE_STATS="True";
export VLLM_DO_NOT_TRACK="True";

echo -e "\\nStarting vLLM API Server with command:";
# Carefully escape the command string for echo to display correctly
ESCAPED_CMD=$(printf '%s' "${vllmServeCommand}" | sed 's/\\$/\\\\\\\\/g') # Escape trailing backslashes for echo -e
echo -e "\${ESCAPED_CMD}\\n"; 

# Execute vLLM Server using exec (replaces shell process)
exec ${vllmServeCommand}
# The job will end when vLLM terminates. Output goes to Slurm files.

# If not using exec:
# nohup ${vllmServeCommand} > "${SCRIPTS_DIR_JS}/${safeJobName}_\${SLURM_JOB_ID}_vllm_stdout.log" 2>&1 &
# VLLM_PID=\$! ; echo "vLLM PID: \${VLLM_PID}" ; sleep 5
# echo "Endpoint Info..." # Add echo block here if needed
# wait \${VLLM_PID} ; EXIT_CODE=\$?
# echo "Job End (vLLM Exit: \${EXIT_CODE})"

`; // End of sbatchScriptContent template literal

    // --- Display Script and Instructions ---
    const deployScriptName = `deploy_${safeJobName}.slurm`; // Use safe name
    const scriptPath = `${SCRIPTS_DIR_JS}/${deployScriptName}`;

    // Using template literal for the final output message
    const finalOutput = `Deployment script for '${selectedModel}' generated.
Save as: ${scriptPath}

-------------------------------------
${sbatchScriptContent}
-------------------------------------

To run the deployment:
1. Save the content above into the file: ${scriptPath}
2. Make it executable:
   chmod +x "${scriptPath}"
3. Submit to Slurm:
   sbatch "${scriptPath}"

Check Slurm output file (e.g., ${SCRIPTS_DIR_JS}/${safeJobName}_<job_id>.out) for endpoint details and logs.
The endpoint will typically be http://<allocated_slurm_node_hostname>:${servicePort}/docs
`;
    outputDiv.textContent = warning + finalOutput; // Prepend warning if any
    console.log("SERAPHIM_DEBUG: sbatch script content generated and displayed.");
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("SERAPHIM_DEBUG: DOM fully loaded and parsed.");
    try {
        populateStaticModels(); 
        
        const deployBtn = document.getElementById('deploy-button');
        const refreshBtn = document.getElementById('refresh-endpoints-button');
        
        if(deployBtn) { 
            deployBtn.addEventListener('click', handleDeployClick); 
            console.log("SERAPHIM_DEBUG: Deploy button event listener added."); 
        } else { 
            console.error("SERAPHIM_DEBUG: Deploy button not found!"); 
        };
        
        if(refreshBtn) { 
            refreshBtn.addEventListener('click', refreshDeployedEndpoints); 
            console.log("SERAPHIM_DEBUG: Refresh endpoints button event listener added."); 
        } else { 
            console.error("SERAPHIM_DEBUG: Refresh endpoints button not found!"); 
        };
    } catch (err) {
        console.error("SERAPHIM_DEBUG: Error during initial setup:", err);
        const outputDiv = document.getElementById('output');
        if(outputDiv) outputDiv.textContent = "An error occurred during page initialization. Check console (F12).";
    }
}); // End DOMContentLoaded listener

EOF
# End of JavaScript file generation
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write JavaScript file."; exit 1; fi;

# Replace placeholders in the generated JavaScript file
# Using a simple sed approach that should be safe for these specific placeholders
ESCAPED_CONDA_ENV_NAME_FOR_SED=$(printf '%s\n' "$CONDA_ENV_NAME" | sed 's:[&/\]:\\&:g');
ESCAPED_SCRIPTS_DIR_FOR_SED=$(printf '%s\n' "$SCRIPTS_DIR" | sed 's:[&/\]:\\&:g');

sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$JS_TARGET_PATH";
if [ $? -ne 0 ]; then echo "❌ Error: Failed sed replace for CONDA_ENV_NAME."; exit 1; fi;
sed -i "s|{{SCRIPTS_DIR_PLACEHOLDER}}|$ESCAPED_SCRIPTS_DIR_FOR_SED|g" "$JS_TARGET_PATH";
if [ $? -ne 0 ]; then echo "❌ Error: Failed sed replace for SCRIPTS_DIR."; exit 1; fi;

echo "✅ JavaScript logic file ($JS_FILENAME) created and configured in $SERAPHIM_DIR.";
echo "";


# --- Generate Minimal HTML File ---
echo "Generating HTML file: $HTML_TARGET_PATH";
# Use cat with 'EOF' - this HTML is now much simpler
cat > "$HTML_TARGET_PATH" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>✧ SERAPHIM CORE ✧ vLLM Deployment Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #029702; /* Electric blue */
            --secondary-color: #f3cb00; /* Lighter cyan/blue */
            --accent-color: #ffcc00; /* Gold accent */
            --bg-color: #1c1c1e; /* Dark background */
            --card-bg-color: #2c2c2e; /* Slightly lighter card background */
            --text-color: #e5e5e7; /* Light text */
            --text-muted-color: #8e8e93;
            --border-color: #3a3a3c;
            --font-body: 'Exo 2', sans-serif;
            --font-heading: 'Orbitron', sans-serif; /* Tech font */
        }
        body { 
            font-family: var(--font-body); 
            margin: 0; padding:0; background-color: var(--bg-color); 
            color: var(--text-color); display: flex; flex-direction: column; 
            min-height: 100vh; font-size: 16px; line-height: 1.6;
        }
        .header { 
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); 
            color: white; padding: 20px 30px; text-align: center; 
            border-bottom: 3px solid var(--accent-color);
            text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        .header h1 { 
            margin: 0; font-family: var(--font-heading); font-size: 2.3em; 
            font-weight: 700; letter-spacing: 2px; display: flex; align-items: center; justify-content: center; gap: 15px;
        }
        .header p { margin: 8px 0 0; font-size: 0.95em; opacity: 0.9; font-weight: 300;}
        .main-container { display: flex; flex-wrap: wrap; padding: 20px; gap: 20px; flex-grow: 1; max-width: 1300px; margin: 20px auto; width: 100%; box-sizing: border-box;}
        .form-container, .endpoints-container { 
            background-color: var(--card-bg-color); padding: 25px; border-radius: 10px; 
            border: 1px solid var(--border-color); box-sizing: border-box;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .form-container { flex: 2; min-width: 400px; }
        .endpoints-container { flex: 1; min-width: 350px; }
        h3 { 
            font-family: var(--font-heading); color: var(--secondary-color); 
            border-bottom: 1px solid var(--accent-color); padding-bottom: 10px; 
            margin-top: 0; margin-bottom: 20px; font-size: 1.4em; letter-spacing: 1px;
            display: flex; align-items: center; gap: 10px;
        }
        label { display: block; margin-top: 15px; margin-bottom: 5px; font-weight: 600; font-size: 0.9em; color: var(--text-muted-color); text-transform: uppercase; letter-spacing: 0.5px;}
        select, input[type="text"], input[type="number"], input[type="email"], input[type="password"] { 
            width: 100%; padding: 12px; margin-bottom: 15px; border-radius: 5px; 
            border: 1px solid var(--border-color); box-sizing: border-box; font-size: 0.95em; 
            background-color: #3a3a3c; color: var(--text-color); 
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        select:focus, input:focus { 
            border-color: var(--primary-color); 
            box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.3); 
            outline: none; background-color: #4a4a4e;
        }
        button { 
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color)); 
            color: white; padding: 12px 20px; cursor: pointer; border: none; border-radius: 5px; 
            font-weight: bold; font-size: 1em; text-transform: uppercase; letter-spacing: 0.8px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s ease; margin-top: 10px;
            width: 100%; /* Make buttons full width */
        }
        button:hover, button:focus { 
            background: linear-gradient(to right, var(--secondary-color), var(--primary-color)); 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3); transform: translateY(-2px); 
        }
        #output { 
            margin-top: 20px; padding: 15px; background-color: #1c1c1e; /* Darker background for output */
            border: 1px solid var(--border-color); border-radius: 6px; white-space: pre-wrap; 
            word-wrap: break-word; font-size: 0.9em; max-height: 400px; overflow-y: auto; 
            line-height: 1.6; font-family: 'Courier New', Courier, monospace; color: #a5d6ff; /* Light blue text for code */
        }
        .slurm-options h3 { margin-top: 25px; font-size: 1.2em;}
        .slurm-options label { font-weight: 400; font-size: 0.85em; margin-top: 8px; }
        
        .endpoints-container #refresh-endpoints-button { 
             background: linear-gradient(to right, #ffcc00, #ff9500); /* Gold/Orange gradient */
             margin-bottom: 15px;
        }
        .endpoints-container #refresh-endpoints-button:hover { 
            background: linear-gradient(to right, #ff9500, #ffcc00); 
        }
        .endpoint-item { 
            background-color: #3a3a3c; border: 1px solid #4a4a4e; padding: 15px; 
            margin-bottom: 10px; border-radius: 6px; font-size: 0.9em; line-height: 1.6; 
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
        }
        .endpoint-item strong { color: var(--secondary-color); } /* Lighter blue */
        .endpoint-item a { color: var(--accent-color); text-decoration: none; font-weight: bold;}
        .endpoint-item a:hover { text-decoration: underline; color: #ffd633; /* Brighter gold */ }
        
        .footer { 
            text-align: center; padding: 20px; background-color: #0e0e0f; /* Even darker footer */
            color: #8e8e93; font-size: 0.9em; margin-top: auto; 
            border-top: 3px solid var(--accent-color); 
        }
        .icon { margin-right: 8px; font-size: 1.2em; vertical-align: middle;} /* For unicode symbols */
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="icon"></span> SERAPHIM <span class="icon"></span></h1>
        <p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p>
    </div>

    <div class="main-container">
        <div class="form-container">
            <h3><span class="icon">⚙️</span> Deploy New vLLM Instance</h3>
            
            <label for="model-select">Select Model:</label>
            <select id="model-select">
                <option value="">-- Select a Model --</option>
            </select>

            <label for="service-port">Service Port (on Slurm node):</label>
            <input type="number" id="service-port" value="8000" min="1024" max="65535"/>

            <label for="hf-token">Hugging Face Token (Optional):</label>
            <input type="password" id="hf-token" placeholder="Needed for Llama, gated models, etc."/>

            <div class="slurm-options">
                <h3>Slurm Configuration</h3>
                <label for="job-name">Job Name:</label>
                <input type="text" id="job-name" value="vllm_service"/>
                <label for="time-limit">Time Limit (HH:MM:SS):</label>
                <input type="text" id="time-limit" value="23:59:59"/>
                <label for="gpus">GPUs (e.g., 1 or a100:1):</label>
                <input type="text" id="gpus" value="1"/>
                <label for="cpus-per-task">CPUs per Task:</label>
                <input type="number" id="cpus-per-task" value="4" min="1"/>
                <label for="mem">Memory (e.g., 32G):</label>
                <input type="text" id="mem" value="32G"/>
                <label for="mail-user">Email Notify (Optional):</label>
                <input type="email" id="mail-user" placeholder="your_email@example.com"/>
            </div>

            <button id="deploy-button">Generate Slurm Script</button>
            <div id="output">Select model & click button to generate script.</div>
        </div>

        <div class="endpoints-container">
             <h3><span class="icon">📡</span> Active Deployments (Placeholder)</h3>
             <button id="refresh-endpoints-button">Refresh Status</button>
             <div id="deployed-endpoints-list">
                 <p>Requires backend integration to list active Slurm jobs.</p>
             </div>
        </div>
    </div>

    <div class="footer">
        ✧ SERAPHIM CORE Interface v1.0 ✧ System Online ✧
    </div>

    <script src="seraphim_logic.js" defer></script> 

</body>
</html>
EOF
# End of HTML file generation
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write HTML file."; exit 1; fi;

echo "✅ Frontend HTML ($HTML_FILENAME) created/updated in $SERAPHIM_DIR.";
echo "";

# --- Final Instructions ---
echo "==========================================";
echo "✅ SERAPHIM Setup Complete!";
echo "==========================================";
echo "";
echo "To use SERAPHIM:";
echo "";
echo "1. Activate Conda Env (in NEW terminal sessions):";
echo "   conda activate $CONDA_ENV_NAME";
echo "";
echo "2. Start HTTP Server (ON SERVER: $HOSTNAME):";
echo "   cd \"$SERAPHIM_DIR\"";
echo "   python -m http.server --bind 0.0.0.0 8869";
echo "";
echo "3. Access in Browser:";
_SERVER_IP=$(hostname -I | awk '{print $1}' || echo "YOUR_SERVER_IP"); # Attempt to get IP
echo "   Open http://${_SERVER_IP}:8869";
echo "";
echo "** Dropdown should work now! Check Browser Console (F12) for errors if problems persist. **";
echo "";
echo "File Locations:";
echo " - HTML Interface: $HTML_TARGET_PATH";
echo " - JavaScript Logic: $JS_TARGET_PATH";
echo " - Slurm Scripts Dir: $SCRIPTS_DIR";
echo "==========================================";

exit 0;