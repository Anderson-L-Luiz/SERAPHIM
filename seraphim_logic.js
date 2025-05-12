// seraphim_logic.js
const BACKEND_API_BASE_URL = `http://${window.location.hostname}:8870/api`;
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
