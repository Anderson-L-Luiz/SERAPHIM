// seraphim_logic.js
const BACKEND_API_URL_DEPLOY = `http://${window.location.hostname}:8870/api/deploy`;
const BACKEND_API_URL_ACTIVE = `http://${window.location.hostname}:8870/api/active_deployments`;
const MODELS_FILE_URL = 'models.txt'; 

let allModels = []; 

async function fetchAndPopulateModels() {
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
            const errorText = `Failed to fetch ${MODELS_FILE_URL}: ${response.status} ${response.statusText}. Please ensure models.txt exists in the SERAPHIM directory.`;
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
                    const modelId = parts[0].trim();
                    const modelName = parts.slice(1).join(',').trim(); 
                    allModels.push({ id: modelId, name: modelName });
                } else if (parts.length === 1 && parts[0]) { 
                    allModels.push({ id: parts[0], name: parts[0] });
                }
            }
        });

        if (allModels.length === 0) {
            modelSelect.innerHTML = '<option value="">-- No models found in models.txt --</option>';
            console.warn(`SERAPHIM_DEBUG: No models parsed from ${MODELS_FILE_URL} or file is empty/incorrectly formatted.`);
            const outputDiv = document.getElementById('output');
            if(outputDiv) outputDiv.textContent = `⚠️ Warning: No models found in ${MODELS_FILE_URL}. Please check the file content and format.`;
            return;
        }
        
        allModels.sort((a, b) => a.name.localeCompare(b.name));
        populateModelDropdown(allModels); 
        console.log(`SERAPHIM_DEBUG: Successfully fetched and parsed ${allModels.length} models from ${MODELS_FILE_URL}.`);

    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        modelSelect.innerHTML = `<option value="">-- Error loading models from ${MODELS_FILE_URL} --</option>`;
        const outputDiv = document.getElementById('output');
        if(outputDiv) outputDiv.textContent = `❌ ${error.message}`;
    }
}

function populateModelDropdown(modelsToDisplay) {
    const modelSelect = document.getElementById('model-select');
    modelSelect.innerHTML = ''; 

    const placeholderOption = document.createElement('option');
    placeholderOption.value = "";
    placeholderOption.textContent = "-- Select a Model --";
    modelSelect.appendChild(placeholderOption);

    modelsToDisplay.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name; 
        modelSelect.appendChild(option);
    });
}

function filterModels() {
    const searchTerm = document.getElementById('model-search').value.toLowerCase();
    const filteredModels = allModels.filter(model => 
        model.name.toLowerCase().includes(searchTerm) || model.id.toLowerCase().includes(searchTerm)
    );
    populateModelDropdown(filteredModels);
}


async function refreshDeployedEndpoints() {
    console.log("SERAPHIM_DEBUG: Refreshing active deployments...");
    const listDiv = document.getElementById('deployed-endpoints-list');
    const refreshButton = document.getElementById('refresh-endpoints-button');
    if (!listDiv || !refreshButton) {
        console.error("SERAPHIM_DEBUG: UI elements for active deployments not found.");
        return;
    }
    listDiv.innerHTML = "<p><em>🔄 Fetching active deployments...</em></p>";
    refreshButton.disabled = true;

    try {
        const response = await fetch(BACKEND_API_URL_ACTIVE);
        if (!response.ok) {
            const errorResult = await response.json().catch(() => ({ detail: "Unknown error fetching active deployments." }));
            throw new Error(`Failed to fetch active deployments: ${response.status} ${response.statusText} - ${errorResult.detail}`);
        }
        const deployments = await response.json();
        
        if (deployments.length === 0) {
            listDiv.innerHTML = "<p>No active SERAPHIM vLLM deployments found for your user.</p>";
        } else {
            let html = '<ul>';
            deployments.forEach(job => {
                html += `<li class="endpoint-item">
                    <strong>Job ID:</strong> ${job.job_id}<br/>
                    <strong>Name:</strong> ${job.job_name || 'N/A'}<br/>
                    <strong>Status:</strong> ${job.status || 'N/A'}<br/>
                    <strong>Node(s):</strong> ${job.nodes || (job.status === 'PD' ? 'Pending Allocation' : 'N/A')}<br/>
                    <strong>User:</strong> ${job.user || 'N/A'}<br/>
                    <strong>Time Used:</strong> ${job.time_used || 'N/A'}<br/>
                    ${job.service_url ? `<strong>Service URL:</strong> <a href="${job.service_url}" target="_blank">${job.service_url}</a><br/>` : (job.status === 'R' ? '<em>Service URL pending log scan...</em><br/>' : '')}
                    ${job.slurm_output_file ? `<strong>Slurm Out:</strong> ${job.slurm_output_file}<br/>` : ''}
                    </li>`;
            });
            html += '</ul>';
            listDiv.innerHTML = html;
        }
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error refreshing active deployments:", error);
        listDiv.innerHTML = `<p style="color: var(--error-color);">❌ Error fetching active deployments: ${error.message}</p>`;
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

    const maxModelLenInput = document.getElementById('max-model-len').value;
    const maxModelLen = maxModelLenInput ? parseInt(maxModelLenInput, 10) : null;

    if (maxModelLenInput && (isNaN(maxModelLen) || maxModelLen <= 0)) {
        outputDiv.textContent = "⚠️ Max Model Length must be a positive number if specified.";
        outputDiv.style.color = "var(--warning-color, #f3cb00)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend"; return;
    }

    const slurmConfig = {
        selected_model: document.getElementById('model-select').value,
        service_port: document.getElementById('service-port').value,
        hf_token: document.getElementById('hf-token').value || null,
        max_model_len: maxModelLen, 
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
    if (!slurmConfig.job_name) { 
        outputDiv.textContent = "⚠️ Please enter a Job Name.";
        outputDiv.style.color = "var(--warning-color, #f3cb00)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend"; return;
    }

    try {
        const response = await fetch(BACKEND_API_URL_DEPLOY, {
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
            refreshDeployedEndpoints(); 
        } else {
            outputDiv.style.color = "var(--error-color, #ff3b30)";
            outputDiv.textContent = `❌ Error (${response.status}): ${result.detail || response.statusText || 'Unknown backend error.'}`;
        }
    } catch (error) { 
        outputDiv.style.color = "var(--error-color, #ff3b30)";
        outputDiv.textContent = `❌ Network/Connection Error or Invalid Response: ${error.message}. Is backend at ${BACKEND_API_URL_DEPLOY} running and returning valid JSON?`;
    } finally {
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchAndPopulateModels(); 
    
    const modelSearchInput = document.getElementById('model-search');
    if (modelSearchInput) {
        modelSearchInput.addEventListener('input', filterModels);
    } else {
        console.error("SERAPHIM_DEBUG: Model search input (#model-search) not found!");
    }

    const deployBtn = document.getElementById('deploy-button');
    if (deployBtn) deployBtn.addEventListener('click', handleDeployClick);
    else console.error("SERAPHIM_DEBUG: Deploy button (#deploy-button) not found!");
    
    const refreshBtn = document.getElementById('refresh-endpoints-button');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshDeployedEndpoints);
    else console.error("SERAPHIM_DEBUG: Refresh button (#refresh-endpoints-button) not found!");
    
    refreshDeployedEndpoints(); 
});
