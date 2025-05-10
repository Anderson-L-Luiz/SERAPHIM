// seraphim_logic.js
const BACKEND_API_URL = `http://${window.location.hostname}:8870/api/deploy`;
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
