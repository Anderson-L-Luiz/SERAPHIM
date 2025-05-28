// seraphim_logic.js
const BACKEND_API_BASE_URL = `http://${window.location.hostname}:8870/api`;
const MODELS_FILE_URL = 'models.txt';

let allModels = [];
let currentSelectedJobDetails = null; 
const LOG_REFRESH_INTERVAL = 500; 

async function fetchAndPopulateModels() {
    console.log("SERAPHIM_DEBUG: Fetching models from", MODELS_FILE_URL);
    const modelSelect = document.getElementById('model-select');
    const modelSearchInput = document.getElementById('model-search'); 
    if (!modelSelect || !modelSearchInput) {
        console.error("SERAPHIM_DEBUG: Model select or search input not found!");
        return;
    }
    
    const initialText = modelSelect.disabled ? '<option value="">-- Model list disabled --</option>' : '<option value="">-- Loading models... --</option>';
    modelSelect.innerHTML = initialText;

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
            if (!modelSelect.disabled) {
                modelSelect.innerHTML = '<option value="">-- No models in models.txt --</option>';
            }
            // Do not overwrite main output here, let user interact first.
            // document.getElementById('output')?.textContent = `⚠️ No models in ${MODELS_FILE_URL}.`;
            console.warn(`SERAPHIM_DEBUG: No models found in ${MODELS_FILE_URL} or file is empty.`);
        }
        allModels.sort((a, b) => a.name.localeCompare(b.name));
        if (!modelSelect.disabled) { // Only populate if the dropdown isn't supposed to be disabled
            populateModelDropdown(allModels);
        }
    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        if (!modelSelect.disabled) {
            modelSelect.innerHTML = `<option value="">-- Error loading models --</option>`;
        }
        // document.getElementById('output')?.textContent = `❌ ${error.message}`;
        console.error(`SERAPHIM_DEBUG: Failed to load models.txt: ${error.message}`);
    }
}

function populateModelDropdown(modelsToDisplay) {
    const modelSelect = document.getElementById('model-select');
    const searchVal = document.getElementById('model-search').value;
    const currentSelectedVal = modelSelect.value;
    modelSelect.innerHTML = ''; 

    if (modelsToDisplay.length === 0 && allModels.length > 0) { // Search yielded no results from a populated list
        const opt = document.createElement('option');
        opt.value = ""; 
        opt.textContent = "-- No models match search --";
        modelSelect.appendChild(opt);
    } else if (allModels.length === 0) { // The models.txt itself was empty or failed to load
       const opt = document.createElement('option');
        opt.value = ""; 
        opt.textContent = "-- Model list empty/unavailable --";
        modelSelect.appendChild(opt);
    }
    else {
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
    else if (!searchVal) modelSelect.value = ""; 
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
             displayElement.dataset.hasContent = "true"; 
             const nearBottom = displayElement.scrollHeight - displayElement.clientHeight - displayElement.scrollTop < 50; 
             if (isInitialFetch || nearBottom) displayElement.scrollTop = displayElement.scrollHeight;
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
            if (!currentSelectedJobDetails) { 
                logOutDisplay.textContent = "No active jobs."; logOutDisplay.dataset.hasContent = "false";
                logErrDisplay.textContent = "No active jobs."; logErrDisplay.dataset.hasContent = "false";
            }
            currentSelectedJobDetails = null;
        } else {
            let html = '<ul>';
            deployments.forEach(job => {
                const outFile = job.slurm_output_file || '';
                const errFile = job.slurm_error_file || '';
                
                // START: MODIFIED ACCESS LINK LOGIC
                let accessLine = '';
                const isJobRunning = job.status === 'R'; // Assuming 'R' is the primary status for running from squeue

                if (isJobRunning && job.detected_port) {
                    const port = job.detected_port;
                    const displayAccessText = `10.16.246.2:${port}/docs`; // As per user request
                    let actualDocsHref = '';

                    if (job.service_url) { // Prefer fully resolved service_url from backend
                        let baseServiceUrl = job.service_url;
                        // Remove /docs if already present, and trailing slash, before appending /docs
                        if (baseServiceUrl.endsWith('/docs')) {
                             baseServiceUrl = baseServiceUrl.substring(0, baseServiceUrl.length - '/docs'.length);
                        }
                        if (baseServiceUrl.endsWith('/')) {
                             baseServiceUrl = baseServiceUrl.substring(0, baseServiceUrl.length - 1);
                        }
                        actualDocsHref = `${baseServiceUrl}/docs`;
                    } else if (job.node_ip) { // Fallback: construct from node_ip if service_url isn't parsed/ready yet
                        const host = job.node_ip.replace(/^https?:\/\//, ''); // Clean node_ip just in case it has scheme
                        actualDocsHref = `http://${host}:${port}/docs`;
                    }

                    if (actualDocsHref) {
                        accessLine = `<br/><strong>Access:</strong> <a href="${actualDocsHref}" target="_blank" onclick="event.stopPropagation();">${displayAccessText}</a>`;
                    } else {
                        // This case means job is running, port detected, but no node_ip/service_url to form a Href
                        accessLine = `<br/><strong>Access:</strong> ${displayAccessText} (Link endpoint info missing)`;
                    }
                } else if (isJobRunning && job.node_ip) { // Job is Running, has node_ip, but port not yet detected
                    accessLine = `<br/><em>Service on ${job.node_ip} (Port: awaiting detection for /docs link)</em>`;
                } else if (isJobRunning) { // Job is Running, but no node_ip or port details yet
                    accessLine = `<br/><em>Service is running (Details pending for /docs link...)</em>`;
                }
                // END: MODIFIED ACCESS LINK LOGIC
                
                html += `<li class="endpoint-item" data-jobid="${job.job_id}" data-outfile="${outFile}" data-errfile="${errFile}">
                    <strong>Job ID:</strong> ${job.job_id} (${job.status || 'N/A'})<br/>
                    <strong>Name:</strong> ${job.job_name || 'N/A'}
                    ${job.nodes ? `<br/><strong>Node(s):</strong> ${job.nodes}` : ''}
                    ${job.node_ip ? `<br/><strong>Node IP:</strong> ${job.node_ip}` : ''}
                    ${accessLine}
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
                    logOutDisplay.dataset.hasContent = "false"; logErrDisplay.dataset.hasContent = "false"; 
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
                    itemToSelect.click(); 
                    jobAutoSelectedViaParams = true;
                } else {
                     console.warn(`SERAPHIM_DEBUG: Auto-select: job ${jobToSelect.jobId} not found in list.`);
                }
            }
            
            if (!jobAutoSelectedViaParams && currentSelectedJobDetails) {
                const itemToReselect = listDiv.querySelector(`.endpoint-item[data-jobid="${currentSelectedJobDetails.jobId}"]`);
                if (itemToReselect) {
                    itemToReselect.classList.add('selected'); 
                } else { 
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
    deployButton.disabled = true;
    deployButton.textContent = "Submitting...";
    outputDiv.textContent = "🚀 Submitting deployment request...";
    outputDiv.style.color = "var(--text-color)";

    // --- Model Selection Logic ( 그대로 유지 ) ---
    const modelSourceCustomRadio = document.getElementById('model-source-custom');
    let selectedModelIdentifier;

    if (modelSourceCustomRadio.checked) {
        selectedModelIdentifier = document.getElementById('custom-model-path').value.trim();
        if (!selectedModelIdentifier) {
            outputDiv.textContent = "⚠️ Please enter the Custom Local Model Path.";
            outputDiv.style.color = "var(--warning-color)";
            deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
        }
        if (!selectedModelIdentifier.startsWith('/')) { // Basic check for absolute path
            outputDiv.textContent = "⚠️ Custom Local Model Path must be an absolute path (e.g., starting with '/').";
            outputDiv.style.color = "var(--warning-color)";
            deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
        }
    } else { // Model source is list
        selectedModelIdentifier = document.getElementById('model-select').value;
        if (!selectedModelIdentifier) { // Assuming model IDs don't need trimming if populated correctly
            outputDiv.textContent = "⚠️ Please select a model from the list.";
            outputDiv.style.color = "var(--warning-color)";
            deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
        }
    }
    // --- End of Model Selection Logic ---

    // --- Get and Validate Form Inputs ---
    const servicePortValue = document.getElementById('service-port').value.trim();
    const jobNameInputValue = document.getElementById('job-name').value.trim();

    // Validate that Job Name and Service Port are not empty
    if (!jobNameInputValue || !servicePortValue) {
        outputDiv.textContent = "⚠️ Please complete all required fields: Job Name and Service Port.";
        outputDiv.style.color = "var(--warning-color)";
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
    }
    // Note: HTML5 input type="number" with min/max attributes provides some client-side validation for servicePortValue.
    // Additional JavaScript validation for port range (e.g., 1024-65535) can be added here if desired.

    // Construct final job name with the correct port suffix
    // 1. Remove any existing _p<digits> suffix from the user's job name input
    let processedJobName = jobNameInputValue.replace(/_p\d{4,5}$/, '');
    // 2. Append the new _p<PORT> suffix using the value from the "Service Port" field
    processedJobName = `${processedJobName}_p${servicePortValue}`;

    const maxModelLenStr = document.getElementById('max-model-len').value.trim();
    let maxModelLenParsed = null;
    if (maxModelLenStr) { // Only parse if a value is provided
        maxModelLenParsed = parseInt(maxModelLenStr, 10);
        if (isNaN(maxModelLenParsed) || maxModelLenParsed <= 0) {
            outputDiv.textContent = "⚠️ Max Model Length must be a positive number if specified.";
            outputDiv.style.color = "var(--warning-color)";
            deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm"; return;
        }
    }
    // --- End of Input Validation and Processing ---

    const slurmConfig = {
        selected_model: selectedModelIdentifier,
        service_port: servicePortValue, // This is the actual port the service will listen on
        hf_token: document.getElementById('hf-token').value.trim() || null,
        max_model_len: maxModelLenParsed,
        job_name: processedJobName, // Use the processed job name with the correct port suffix
        time_limit: document.getElementById('time-limit').value.trim(),
        gpus: document.getElementById('gpus').value.trim(),
        cpus_per_task: document.getElementById('cpus-per-task').value.trim(), // HTML input type="number"
        mem: document.getElementById('mem').value.trim(),
    };

    // --- API Call Logic ( 그대로 유지 ) ---
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
        await refreshDeployedEndpoints(jobToSelectParams); // Auto-select and show new job's logs
        
        // Optional: Add a slight delay and refresh again to catch details that might populate slowly from Slurm
        setTimeout(async () => {
            console.log("SERAPHIM_DEBUG: Attempting 3-second follow-up refresh for new job details.");
            await refreshDeployedEndpoints(jobToSelectParams); 
        }, 3000);

    } catch (error) {
        outputDiv.style.color = "var(--error-color)";
        outputDiv.textContent = `❌ Error: ${error.message}`;
    } finally {
        deployButton.disabled = false;
        deployButton.textContent = "Deploy to Slurm";
    }
    // --- End of API Call Logic ---
}

async function pollCurrentJobLogs() {
    if (currentSelectedJobDetails && currentSelectedJobDetails.jobId) {
        if (currentSelectedJobDetails.outFile && currentSelectedJobDetails.outFile !== 'null' && currentSelectedJobDetails.outFile !== 'undefined') {
            await fetchLogContent(currentSelectedJobDetails.outFile, 'log-output-content');
        }
        if (currentSelectedJobDetails.errFile && currentSelectedJobDetails.errFile !== 'null' && currentSelectedJobDetails.errFile !== 'undefined') {
            await fetchLogContent(currentSelectedJobDetails.errFile, 'log-error-content');
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    // ---------- Start: New Model Source Toggle Logic ----------
    const modelSourceListRadio = document.getElementById('model-source-list');
    const modelSourceCustomRadio = document.getElementById('model-source-custom');
    const modelListSelectionContainer = document.getElementById('model-list-selection-container');
    const customModelPathContainer = document.getElementById('custom-model-path-container');
    const modelSearchInput = document.getElementById('model-search'); // Already declared for filterModels
    const modelSelectDropdown = document.getElementById('model-select'); // Already declared
    const customModelPathInput = document.getElementById('custom-model-path');

    function updateModelSourceView() {
        if (modelSourceCustomRadio.checked) {
            modelListSelectionContainer.style.display = 'none';
            customModelPathContainer.style.display = 'block';
            modelSearchInput.disabled = true;
            modelSelectDropdown.disabled = true;
            customModelPathInput.disabled = false;
        } else { // modelSourceListRadio.checked (default)
            modelListSelectionContainer.style.display = 'block';
            customModelPathContainer.style.display = 'none';
            modelSearchInput.disabled = false;
            modelSelectDropdown.disabled = false;
            customModelPathInput.disabled = true;
        }
    }

    modelSourceListRadio.addEventListener('change', updateModelSourceView);
    modelSourceCustomRadio.addEventListener('change', updateModelSourceView);
    // ---------- End: New Model Source Toggle Logic ----------
    
    await fetchAndPopulateModels(); 
    document.getElementById('model-search')?.addEventListener('input', filterModels);
    document.getElementById('deploy-button')?.addEventListener('click', handleDeployClick);
    document.getElementById('refresh-endpoints-button')?.addEventListener('click', () => refreshDeployedEndpoints());
    
    updateModelSourceView(); // Call to set initial state based on default radio 'checked'
    await refreshDeployedEndpoints(); 
    setInterval(pollCurrentJobLogs, LOG_REFRESH_INTERVAL); 
});
