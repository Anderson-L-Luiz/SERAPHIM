// --- Constants ---
// These placeholders will be replaced by sed in install.sh
const CONDA_ENV_NAME_JS = "seraphim_vllm_env";
const SCRIPTS_DIR_JS = "/home/aimotion_api/SERAPHIM/scripts"; // This is $SERAPHIM_DIR/scripts

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
    listDiv.innerHTML = "<p><em>Refreshing... (Backend integration with 'squeue' or similar needed to list actual Slurm jobs and parse their vLLM log files.)</em></p>";
    // Example of how an endpoint might be displayed (static for now)
    const exampleHtml = `
        <p>Requires backend integration with Slurm (e.g., via 'squeue' and parsing vLLM logs).</p>
        <div class="endpoint-item">
            <strong>Model:</strong> example/Sample-7B-Demo<br/>
            <strong>Job ID:</strong> 12345 (Example)<br/>
            <strong>Slurm Out:</strong> ${SCRIPTS_DIR_JS}/vllm_seraph_job_12345.out (Example)<br/>
            <strong>vLLM Log:</strong> ${SCRIPTS_DIR_JS}/vllm_logs/vllm_seraph_job_12345_vllm_service.log (Example)<br/>
            <strong>Node:</strong> cnode01.example (Example)<br/>
            <strong>Port:</strong> 8000 (Example)<br/>
            <strong>Access:</strong> <a href="#" onclick="alert('This is a static example. Actual link would depend on the deployed node and port found in logs.'); return false;">http://cnode01.example:8000/docs</a>
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

    let warning = "";
    if (!hfToken && (selectedModel.toLowerCase().includes("llama") || selectedModel.toLowerCase().includes("meta-llama") || selectedModel.toLowerCase().includes("gated"))) {
        warning = "ℹ️ This model might be gated. Provide an HF Token if needed for access.\\n\\n";
    }
    outputDiv.textContent = warning + "Generating deployment script...";

    // --- Build vLLM Serve Command ---
    // Use the official 'vllm serve' command
    let vllmServeCommand = 'vllm serve';
    // The model ID is the first positional argument
    vllmServeCommand += ' "' + shellEscape(selectedModel) + '"'; // Add selected model, quoted and escaped

    // Add other arguments as named options
    let modelArgs = [
        '--host "0.0.0.0"', // Listen on all interfaces within the Slurm job's allocated node
        '--port ' + shellEscape(servicePort),
        // '--disable-log-stats', // This specific flag might be deprecated. vLLM logs basic info by default.
                                // Check 'vllm serve --help' if you need fine-grained log control.
        '--trust-remote-code' // Necessary for many Hugging Face models
    ];

    let maxModelLen = 16384; // Default
    if (selectedModel.toLowerCase().includes("mixtral")) {
        maxModelLen = 32768;
    }
    // Add other model-specific arguments if needed
    if (selectedModel.toLowerCase().includes("pixtral")) {
        // modelArgs.push('--some-pixtral-specific-arg value');
    }
    modelArgs.push('--max-model-len ' + maxModelLen);
    // Example: Add GPU memory utilization if desired
    // modelArgs.push('--gpu-memory-utilization 0.90');
    // Example: Specify dtype if needed, though 'auto' is often fine
    // modelArgs.push('--dtype auto');

    // Join arguments, ensuring proper line continuation for shell readability in the script
    vllmServeCommand += " " + modelArgs.join(" \\\n    ");


    // --- Build Slurm Script ---
    const safeJobName = shellEscape(jobName);
    const safeTimeLimit = shellEscape(timeLimit);
    const safeGpus = shellEscape(gpus);
    const safeCpusPerTask = shellEscape(cpusPerTask);
    const safeMem = shellEscape(mem);
    const safeMailUser = mailUser ? shellEscape(mailUser) : "";
    const safeSelectedModel = shellEscape(selectedModel); // For display
    const safeHfToken = shellEscape(hfToken); // For export

    const mailTypeLine = safeMailUser ? '#SBATCH --mail-type=ALL\\n#SBATCH --mail-user=' + safeMailUser : "#SBATCH --mail-type=NONE";
    const condaBasePathForSlurm = '$(conda info --base)';
    const vllmLogDir = `${SCRIPTS_DIR_JS}/vllm_logs`; // Path for vLLM service specific logs

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
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: \${SLURM_JOB_ID} running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output/Error files will be in: ${SCRIPTS_DIR_JS}/"
echo "Model: ${safeSelectedModel}"
echo "Target Service Port: ${servicePort}"
echo "Conda Env: ${CONDA_ENV_NAME_JS}"
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

# Set vLLM Env Vars (Optional, check vLLM docs for current recommendations)
# export VLLM_USE_TRITON_FLASH_ATTN="True"; # May or may not be needed depending on vLLM version & xformers
export VLLM_CONFIGURE_LOGGING="0"; # Can be useful, or manage logging via vLLM's own params if preferred
export VLLM_NO_USAGE_STATS="True";
export VLLM_DO_NOT_TRACK="True";

# Create directory for vLLM service logs if it doesn't exist
VLLM_LOG_DIR_SLURM="${vllmLogDir}" # Use the variable defined above in JS
mkdir -p "\${VLLM_LOG_DIR_SLURM}"
VLLM_LOG_FILE="\${VLLM_LOG_DIR_SLURM}/${safeJobName}_\${SLURM_JOB_ID}_vllm_service.log"

echo -e "\\nStarting vLLM API Server in the background..."
echo "Command to be executed:"
# Carefully escape the command string for echo to display correctly
ESCAPED_CMD_FOR_ECHO=$(printf '%s' "${vllmServeCommand}" | sed 's/\\$/\\\\\\\\/g; s/"/\\"/g')
echo -e "\${ESCAPED_CMD_FOR_ECHO}"
echo "vLLM service output (including API endpoint) will be logged to: \${VLLM_LOG_FILE}"

# Execute vLLM Server in the background using nohup
nohup ${vllmServeCommand} > "\${VLLM_LOG_FILE}" 2>&1 &
VLLM_PID=\$!
echo "vLLM Service potentially started with PID: \${VLLM_PID}. Waiting a few seconds for initialization..."
sleep 15 # Give the server more time to start up and log initial messages, especially for larger models.

# Check if the process is still running
if kill -0 \${VLLM_PID} > /dev/null 2>&1; then
    echo "vLLM process \${VLLM_PID} appears to be running."
    echo "Initial logs should be available in \${VLLM_LOG_FILE}"
else
    echo "ERROR: vLLM process \${VLLM_PID} does not seem to be running. Check \${VLLM_LOG_FILE} for errors."
fi

echo ""
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SERVICE STATUS ✝"
echo "Job ID: \${SLURM_JOB_ID} on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Model: ${safeSelectedModel}"
echo "Configured Service Port: ${servicePort}"
echo "vLLM Service Log File: \${VLLM_LOG_FILE}"
echo "---"
echo "The vLLM service is running in the background (PID: \${VLLM_PID})."
echo "To find the API endpoint:"
echo "  1. Check the vLLM log: cat \${VLLM_LOG_FILE}"
echo "  2. Look for lines like 'Uvicorn running on http://<ip_address>:${servicePort}'"
echo "  3. The API will be accessible on the node $(hostname -s) at that address."
echo "     Typically: http://$(hostname -s):${servicePort}"
echo "     OpenAI API Docs (Swagger UI): http://$(hostname -s):${servicePort}/docs"
echo ""
echo "To stop the service manually (on the node $(hostname -s)):"
echo "  kill \${VLLM_PID}"
echo "  (Or use 'scancel \${SLURM_JOB_ID}' if you want Slurm to handle termination signals, though nohup detaches it)"
echo "=================================================================="

echo "Slurm script finished its main tasks. The vLLM service (PID: \${VLLM_PID}) should continue in the background."
# The Slurm job itself will end here, but the nohup'd vLLM process continues.
`; // End of sbatchScriptContent template literal

    const deployScriptName = `deploy_${safeJobName}.slurm`;
    const scriptPath = `${SCRIPTS_DIR_JS}/${deployScriptName}`; // SCRIPTS_DIR_JS is $SERAPHIM_DIR/scripts

    const finalOutput = `Deployment script for '${selectedModel}' generated.
Save as: ${scriptPath}

-------------------------------------
${sbatchScriptContent}
-------------------------------------

To run the deployment:
1. Ensure the Slurm script content above is saved into the file:
   ${scriptPath}
   (The web UI would ideally handle this save and execution step via a backend call)

2. Make it executable (if saving manually):
   chmod +x "${scriptPath}"

3. Submit to Slurm:
   sbatch "${scriptPath}"

4. Monitor the Slurm output file for initial job status (e.g., ${SCRIPTS_DIR_JS}/${safeJobName}_<job_id>.out).

5. Check the vLLM service log file for the API endpoint and detailed logs:
   ${vllmLogDir}/${safeJobName}_<job_id>_vllm_service.log
   (The <job_id> will be assigned by Slurm)

The endpoint will typically be http://<allocated_slurm_node_hostname>:${servicePort}/docs
(Replace <allocated_slurm_node_hostname> with the actual node assigned by Slurm, which is printed in the logs)
`;
    outputDiv.textContent = warning + finalOutput;
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

