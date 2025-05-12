// seraphim_logic.js
const BACKEND_API_BASE_URL = `http://${window.location.hostname}:8870/api`;
const MODELS_FILE_URL = 'models.txt';

let allModels = [];
let currentSelectedJobDetails = null; 
// LOG_REFRESH_INTERVAL: 500ms = 0.5 second. 
// This is very frequent and can increase server load significantly.
// Consider increasing if performance issues arise. 1000-2000ms is often a good balance.
const LOG_REFRESH_INTERVAL = 500; 

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
        if (!response.ok) throw new Error(`Failed to fetch ${MODELS_FILE_URL}: ${response.status} ${response.statusText}`);
        const text = await response.text();
        allModels = text.split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('#'))
            .map(line => {
                const parts = line.split(',');
                return parts.length >= 2 ? { id: parts[0].trim(), name: parts.slice(1).join(',').trim() } : { id: line, name: line };
            });

        if (allModels.length === 0) {
            modelSelect.innerHTML = '<option value="">-- No models in models.txt --</option>';
            document.getElementById('output')?.textContent = `⚠️ No models in ${MODELS_FILE_URL}.`;
            return;
        }
        allModels.sort((a, b) => a.name.localeCompare(b.name));
        populateModelDropdown(allModels);
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        modelSelect.innerHTML = `<option value="">-- Error loading models --</option>`;
        document.getElementById('output')?.textContent = `❌ ${error.message}`;
    }
}

function populateModelDropdown(modelsToDisplay) {
    const modelSelect = document.getElementById('model-select');
    const searchVal = document.getElementById('model-search').value;
    const currentSelectedVal = modelSelect.value;
    modelSelect.innerHTML = ''; 

    if (modelsToDisplay.length === 0) {
        const opt = document.createElement('option');
        opt.value = ""; 
        opt.textContent = searchVal ? "-- No models match search --" : "-- Select a Model --";
        modelSelect.appendChild(opt);
    } else {
         // Add a placeholder if search is empty, or always add it
        if (!searchVal) {
            const placeholder = document.createElement('option');
            placeholder.value = ""; 
            placeholder.textContent = "-- Select a Model --";
            modelSelect.appendChild(placeholder);
        }
        modelsToDisplay.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id; option.textContent = model.name;
            modelSelect.appendChild(option);
        });
    }
    if (modelsToDisplay.some(m => m.id === currentSelectedVal)) modelSelect.value = currentSelectedVal;
    else if (!searchVal) modelSelect.value = ""; // Default to placeholder if search empty and previous val gone
}

function filterModels() {
    const searchTerm = document.getElementById('model-search').value.toLowerCase();
    const filtered = allModels.filter(m => m.name.toLowerCase().includes(searchTerm) || m.id.toLowerCase().includes(searchTerm));
    populateModelDropdown(filtered);
}

async function fetchLogContent(filePath, displayElementId) {
    const displayElement = document.getElementById(displayElementId);
    if (!filePath || filePath === 'null' || filePath === 'undefined') {
        displayElement.textContent = 'Log file path not available.'; return;
    }
    const isInitialFetch = !displayElement.dataset.hasContent || displayElement.textContent.startsWith('🔄');
    if (isInitialFetch) displayElement.textContent = `🔄 Fetching ${filePath.split('/').pop()}...`;
    
    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/log_content?file_path=${encodeURIComponent(filePath)}`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP error ${response.status}`);
        
        const newContent = result.log_content || '(Log file is empty)';
        if (displayElement.textContent !== newContent) {
             displayElement.textContent = newContent;
             displayElement.dataset.hasContent = "true"; // Mark that content has been loaded
             constใกล้Bottom = displayElement.scrollHeight - displayElement.clientHeight - displayElement.scrollTop < 50; // Check if user is near bottom
             if (isInitialFetch ||ใกล้Bottom) displayElement.scrollTop = displayElement.scrollHeight;
        }
    } catch (error) {
        console.error(`SERAPHIM_DEBUG: Error fetching log ${filePath}:`, error);
        if (isInitialFetch) displayElement.textContent = `❌ Error fetching log: ${error.message}`;
    }
}

async function cancelJob(jobId) {
    const outputDiv = document.getElementById('output');
    if (!confirm("Are you sure you want to cancel this job? (Be mindful that other users may be using it, consult with your colleagues before proceeding)")) {
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
            document.getElementById('log-output-content').textContent = "Cancelled job's logs cleared.";
            document.getElementById('log-error-content').textContent = "";
            document.querySelectorAll('.endpoint-item.selected').forEach(sel => sel.classList.remove('selected'));
        }
    } catch (error) {
        outputDiv.textContent = `❌ Error cancelling job ${jobId}: ${error.message}`;
        outputDiv.style.color = "var(--error-color)";
    } finally {
        await refreshDeployedEndpoints(); // Refresh list to reflect cancellation
    }
}

async function refreshDeployedEndpoints(jobToSelect = null) {
    console.log("SERAPHIM_DEBUG: Refreshing active deployments...");
    const listDiv = document.getElementById('deployed-endpoints-list');
    const refreshButton = document.getElementById('refresh-endpoints-button');
    const logOutDisplay = document.getElementById('log-output-content');
    const logErrDisplay = document.getElementById('log-error-content');

    if (!listDiv || !refreshButton || !logOutDisplay || !logErrDisplay) return;

    listDiv.innerHTML = "<p><em>🔄 Fetching active jobs...</em></p>";
    if (!jobToSelect && !currentSelectedJobDetails) { 
        currentSelectedJobDetails = null;
        logOutDisplay.textContent = "Select a job to view its API service log.";
        logOutDisplay.dataset.hasContent = "false";
        logErrDisplay.textContent = "Select a job to view its internal vLLM engine log.";
        logErrDisplay.dataset.hasContent = "false";
    }
    refreshButton.disabled = true;

    try {
        const response = await fetch(`${BACKEND_API_BASE_URL}/active_deployments`);
        if (!response.ok) {
          const errRes = await response.json().catch(()=>({detail: "Unknown fetch error"}));
          throw new Error(errRes.detail || `HTTP Error ${response.status}`);
        }
        const deployments = await response.json();

        if (deployments.length === 0) {
            listDiv.innerHTML = "<p>No active Slurm jobs found for your user.</p>";
            if (!currentSelectedJobDetails) { // Only clear if nothing was meant to be selected
                logOutDisplay.textContent = "No active jobs."; logOutDisplay.dataset.hasContent = "false";
                logErrDisplay.textContent = "No active jobs."; logErrDisplay.dataset.hasContent = "false";
            }
            currentSelectedJobDetails = null;
        } else {
            let html = '<ul>';
            deployments.forEach(job => {
                const outFile = job.slurm_output_file || '';
                const errFile = job.slurm_error_file || '';
                let urlDisplay = job.service_url ? job.service_url.replace(/^https?:\/\//, '') : (job.status === 'R' && job.node_ip && job.detected_port ? `${job.node_ip}:${job.detected_port}` : '');
                
                html += `<li class="endpoint-item" data-jobid="${job.job_id}" data-outfile="${outFile}" data-errfile="${errFile}">
                    <strong>Job ID:</strong> ${job.job_id} (${job.status || 'N/A'})<br/>
                    <strong>Name:</strong> ${job.job_name || 'N/A'}
                    ${job.nodes ? `<br/><strong>Node(s):</strong> ${job.nodes}` : ''}
                    ${job.node_ip ? `<br/><strong>Node IP:</strong> ${job.node_ip}` : ''}
                    ${job.service_url ? `<br/><strong>Access:</strong> <a href="${job.service_url}" target="_blank" onclick="event.stopPropagation();">${urlDisplay}</a>` : (job.status === 'R' && job.node_ip ? `<br/><em>Service on ${job.node_ip} (Port: ${job.detected_port || 'N/A'})</em>` : '')}
                    <button class="cancel-job-button" data-jobid="${job.job_id}">Cancel</button>
                 </li>`;
            });
            html += '</ul>';
            listDiv.innerHTML = html;

            listDiv.querySelectorAll('.endpoint-item').forEach(item => {
                item.addEventListener('click', async function() {
                    listDiv.querySelectorAll('.endpoint-item.selected').forEach(sel => sel.classList.remove('selected'));
                    this.classList.add('selected');
                    currentSelectedJobDetails = { 
                        jobId: this.dataset.jobid, 
                        outFile: this.dataset.outfile, 
                        errFile: this.dataset.errfile 
                    };
                    logOutDisplay.dataset.hasContent = "false"; logErrDisplay.dataset.hasContent = "false"; // Reset for initial fetch message
                    await fetchLogContent(currentSelectedJobDetails.outFile, 'log-output-content');
                    await fetchLogContent(currentSelectedJobDetails.errFile, 'log-error-content');
                });
            });

            listDiv.querySelectorAll('.cancel-job-button').forEach(button => {
                button.addEventListener('click', e => { e.stopPropagation(); cancelJob(e.target.dataset.jobid); });
            });
            
            let jobAutoSelectedViaParams = false;
            if (jobToSelect && jobToSelect.jobId) {
                const itemToSelect = listDiv.querySelector(`.endpoint-item[data-jobid="${jobToSelect.jobId}"]`);
                if (itemToSelect) {
                    itemToSelect.click(); // Simulate click to trigger selection and log loading
                    jobAutoSelectedViaParams = true;
                } else {
                     console.warn(`SERAPHIM_DEBUG: Auto-select: job ${jobToSelect.jobId} not found in list.`);
                }
            }
            
            // Maintain selection if the currently selected job is still in the list and wasn't just auto-selected
            if (!jobAutoSelectedViaParams && currentSelectedJobDetails) {
                const itemToReselect = listDiv.querySelector(`.endpoint-item[data-jobid="${currentSelectedJobDetails.jobId}"]`);
                if (itemToReselect) {
                    itemToReselect.classList.add('selected'); // Re-apply class if cleared by innerHTML overwrite
                } else { // Previously selected job is gone
                    currentSelectedJobDetails = null;
                    logOutDisplay.textContent = "Previously selected job no longer active."; logOutDisplay.dataset.hasContent = "false";
                    logErrDisplay.textContent = ""; logErrDisplay.dataset.hasContent = "false";
                }
            } else if (!jobAutoSelectedViaParams && !currentSelectedJobDetails && deployments.length > 0) {
                 logOutDisplay.textContent = "Select a job to view its API service log."; logOutDisplay.dataset.hasContent = "false";
                 logErrDisplay.textContent = "Select a job to view its internal vLLM engine log."; logErrDisplay.dataset.hasContent = "false";
            }
        }
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error refreshing active deployments:", error);
        listDiv.innerHTML = `<p style="color: var(--error-color);">❌ Error fetching: ${error.message}</p>`;
        logOutDisplay.textContent = "Error loading deployments."; logOutDisplay.dataset.hasContent = "false";
        logErrDisplay.textContent = ""; logErrDisplay.dataset.hasContent = "false";
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
        service_port: document.getElementById('service-port').value, // Will be "" if not filled
        hf_token: document.getElementById('hf-token').value || null,
        max_model_len: document.getElementById('max-model-len').value ? parseInt(document.getElementById('max-model-len').value, 10) : null,
        job_name: document.getElementById('job-name').value,
        time_limit: document.getElementById('time-limit').value,
        gpus: document.getElementById('gpus').value,
        cpus_per_task: document.getElementById('cpus-per-task').value,
        mem: document.getElementById('mem').value,
        // mail_user removed
    };

    if (!slurmConfig.selected_model || !slurmConfig.job_name || !slurmConfig.service_port) {
        outputDiv.textContent = "⚠️ Please select model, enter Job Name, and specify Service Port.";
        outputDiv.style.color = "var(--warning-color)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
    }
    // ... (rest of validation and deploy logic from v2.2) ...
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
        
        const jobToSelectParams = { 
            jobId: result.job_id, 
            outFile: result.slurm_output_file_pattern, 
            errFile: result.slurm_error_file_pattern 
        };
        await refreshDeployedEndpoints(jobToSelectParams);
        // Attempt a follow-up refresh to catch Slurm updates for the new job
        setTimeout(async () => {
          console.log("SERAPHIM_DEBUG: Attempting 3-second follow-up refresh for new job details.");
          await refreshDeployedEndpoints(jobToSelectParams); 
        }, 3000);

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
    await fetchAndPopulateModels(); 
    document.getElementById('model-search')?.addEventListener('input', filterModels);
    document.getElementById('deploy-button')?.addEventListener('click', handleDeployClick);
    document.getElementById('refresh-endpoints-button')?.addEventListener('click', () => refreshDeployedEndpoints());
    await refreshDeployedEndpoints(); 
    setInterval(pollCurrentJobLogs, LOG_REFRESH_INTERVAL); 
});
