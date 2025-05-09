// --- Constants ---
// These placeholders will be replaced by sed in install.sh
const CONDA_ENV_NAME_JS = "seraphim_vllm_env";
const SCRIPTS_DIR_JS = "/home/aimotion_api/SERAPHIM/scripts";

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
    // For simple cases, just escape common shell metacharacters.
    // A more robust solution might involve single quoting the whole string if possible.
    return String(str).replace(/([\\"'$`!*()&|;<>\s?#])/g, '\\$1');
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

    // Get values
    const selectedModel = modelSelect ? modelSelect.value : '';
    const servicePort = servicePortInput ? servicePortInput.value.trim() : '8000'; // Default port
    const hfToken = hfTokenInput ? hfTokenInput.value.trim() : '';
    const jobName = jobNameInput ? jobNameInput.value.trim() : 'vllm_seraph_job';
    const timeLimit = timeLimitInput ? timeLimitInput.value.trim() : '23:59:59';
    const gpus = gpusInput ? gpusInput.value.trim() : '1';
    const cpusPerTask = cpusPerTaskInput ? cpusPerTaskInput.value.trim() : '4';
    const mem = memInput ? memInput.value.trim() : '16G';
    const mailUser = mailUserInput ? mailUserInput.value.trim() : '';

    // --- Validation ---
    if (!selectedModel) {
        outputDiv.textContent = "⚠️ Please select a model first.";
        return;
    }
    if (!servicePort || !/^\d+$/.test(servicePort) || parseInt(servicePort, 10) < 1024 || parseInt(servicePort, 10) > 65535) {
        outputDiv.textContent = "⚠️ Please enter a valid service port (1024-65535).";
        return;
    }
    // Basic validation for other Slurm parameters can be added here

    let warning = "";
    if (!hfToken && (selectedModel.toLowerCase().includes("llama") || selectedModel.toLowerCase().includes("meta-llama") || selectedModel.toLowerCase().includes("gated"))) {
        warning = "ℹ️ This model might be gated. Provide an HF Token if needed for access.\n\n";
    }
    outputDiv.textContent = warning + "Generating deployment script...";

    // --- Build vLLM Serve Command ---
    // Ensure selectedModel is treated as a single argument, even with spaces
    let vllmServeCommand = `python -m vllm.entrypoints.openai.api_server`; // Using the OpenAI compatible server
    let modelArgs = [
        `--model "${shellEscape(selectedModel)}"`, // Quote and escape the model name
        `--host "0.0.0.0"`,
        `--port ${shellEscape(servicePort)}`,
        `--disable-log-stats`, // Reduces verbosity
        `--trust-remote-code`  // Often needed for custom models
    ];

    let maxModelLen = 16384; // Default
    if (selectedModel.toLowerCase().includes("mixtral")) {
        maxModelLen = 32768;
    }
    // Add other model-specific arguments if needed
    if (selectedModel.toLowerCase().includes("pixtral")) {
        // Pixtral might require specific args, ensure your vLLM version supports them
        // modelArgs.push('--guided-decoding-backend=lm-format-enforcer');
        // modelArgs.push('--limit_mm_per_prompt \'image=8\'');
    }
    modelArgs.push(`--max-model-len ${maxModelLen}`);

    vllmServeCommand += " " + modelArgs.join(" \\\n    "); // Join with spaces and line continuation for readability

    // --- Build Slurm Script ---
    // Escape user-provided values that will go into the shell script
    const safeJobName = shellEscape(jobName);
    const safeTimeLimit = shellEscape(timeLimit);
    const safeGpus = shellEscape(gpus);
    const safeCpusPerTask = shellEscape(cpusPerTask);
    const safeMem = shellEscape(mem);
    const safeMailUser = mailUser ? shellEscape(mailUser) : ""; // Only escape if provided
    const safeSelectedModel = shellEscape(selectedModel); // For display in echo
    const safeHfToken = shellEscape(hfToken); // For export

    const mailTypeLine = safeMailUser ? `#SBATCH --mail-type=ALL\\n#SBATCH --mail-user=${safeMailUser}` : "#SBATCH --mail-type=NONE";
    const condaBasePathForSlurm = "\\$(conda info --base)"; // Escaped $ for Slurm script to execute `conda info --base`

    const sbatchScriptContent = `#!/bin/bash
#SBATCH --job-name=${safeJobName}
#SBATCH --output=${SCRIPTS_DIR_JS}/${safeJobName}_%j.out
#SBATCH --error=${SCRIPTS_DIR_JS}/${safeJobName}_%j.err
#SBATCH --time=${safeTimeLimit}
#SBATCH --gres=gpu:${safeGpus} # e.g., gpu:1 or gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${safeCpusPerTask}
#SBATCH --mem=${safeMem} # e.g., 16G or 64G
${mailTypeLine}

echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job ✝"
echo "Job Start Time: $(date)"
echo "Job ID: \${SLURM_JOB_ID} running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Model: ${safeSelectedModel}"
echo "Service Port: ${servicePort}"
echo "Conda Env: ${CONDA_ENV_NAME_JS}"
echo "Scripts Dir: ${SCRIPTS_DIR_JS}"
echo "=================================================================="

# Source Conda
CONDA_BASE_PATH_SLURM="${condaBasePathForSlurm}" # This will execute: $(conda info --base)
if [ -z "\${CONDA_BASE_PATH_SLURM}" ]; then
    echo "ERROR: Could not determine Conda base path inside Slurm job."
    exit 1
fi
CONDA_SH_PATH="\${CONDA_BASE_PATH_SLURM}/etc/profile.d/conda.sh"

if [ -f "\${CONDA_SH_PATH}" ]; then
    echo "Sourcing Conda from: \${CONDA_SH_PATH}"
    # shellcheck source=/dev/null
    . "\${CONDA_SH_PATH}"
else
    echo "WARN: conda.sh not found at \${CONDA_SH_PATH}. Conda environment might not activate."
fi

echo "Activating Conda Environment: ${CONDA_ENV_NAME_JS}"
conda activate "${CONDA_ENV_NAME_JS}" # Quoted for safety

# Check if activation was successful
if [[ "$CONDA_PREFIX" != *"${CONDA_ENV_NAME_JS}"* ]]; then
    echo "ERROR: Failed to activate conda environment '${CONDA_ENV_NAME_JS}'."
    echo "Current CONDA_PREFIX: $CONDA_PREFIX"
    exit 1
else
    echo "Conda environment '${CONDA_ENV_NAME_JS}' activated. Path: $CONDA_PREFIX"
fi

# Set Hugging Face Token if provided
if [ -n "${safeHfToken}" ]; then
    export HF_TOKEN="${safeHfToken}"
    echo "HF_TOKEN has been set."
else
    echo "HF_TOKEN not provided or empty."
fi

# vLLM specific environment variables (optional, defaults are usually fine)
export VLLM_USE_TRITON_FLASH_ATTN="True" # If you have flash-attn and want to use it with Triton
export VLLM_CONFIGURE_LOGGING="0" # 0 for basic, 1 for detailed JSON
export VLLM_NO_USAGE_STATS="True" # Disable vLLM usage statistics reporting
export VLLM_DO_NOT_TRACK="True"   # Older alias for VLLM_NO_USAGE_STATS

echo -e "\\nStarting vLLM API Server with command:"
echo -e "${vllmServeCommand}\\n"

# Execute the vLLM server command
exec ${vllmServeCommand}
`; // End of sbatchScriptContent template literal

    // --- Display Script and Instructions ---
    const deployScriptName = `deploy_vllm_job_${safeJobName}.slurm`; // CORRECTED
    const scriptPath = `${SCRIPTS_DIR_JS}/${deployScriptName}`;     // CORRECTED

    let currentOutputContent = outputDiv.textContent.startsWith("ℹ️") ? outputDiv.textContent + "\n\n" : ""; // CORRECTED for newline

    currentOutputContent += `Deployment script for '${selectedModel}' generated.
Save as: ${scriptPath}

-------------------------------------
${sbatchScriptContent}
-------------------------------------

To run the deployment:
1. Save the content above into the file: ${scriptPath}
    (Or this script will attempt to save it if backend functionality were present)
2. Make it executable:
    chmod +x "${scriptPath}"
3. Submit to Slurm:
    sbatch "${scriptPath}"

Check Slurm output file (e.g., ${SCRIPTS_DIR_JS}/${safeJobName}_<job_id>.out) for endpoint details and logs.
The endpoint will typically be http://<allocated_slurm_node_hostname>:${servicePort}/docs
`;
    outputDiv.textContent = currentOutputContent;
    console.log("SERAPHIM_DEBUG: sbatch script content generated and displayed.");

    // Note: Actual file saving would require backend/server-side capabilities or browser download APIs.
    // This script only displays the content for the user to copy.
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
            console.error("SERAPHIM_DEBUG: Deploy button not found in the DOM!");
        }

        if(refreshBtn) {
            refreshBtn.addEventListener('click', refreshDeployedEndpoints);
            console.log("SERAPHIM_DEBUG: Refresh endpoints button event listener added.");
        } else {
            console.error("SERAPHIM_DEBUG: Refresh endpoints button not found in the DOM!");
        }
    } catch (err) {
        console.error("SERAPHIM_DEBUG: Error during initial setup in DOMContentLoaded:", err);
        const outputDiv = document.getElementById('output');
        if(outputDiv) outputDiv.textContent = "An error occurred during page initialization. Check console (F12).";
    }
});
