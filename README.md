# SERAPHIM: Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling

**Version:** 1.3 (Foreground Slurm Service with Backend Integration)
**Author:** Anderson de Lima Luiz

## Overview

SERAPHIM provides a web-based interface to simplify the deployment of vLLM (Very Large Language Model) services as Slurm batch jobs. It allows users to select from a predefined list of models, configure Slurm resource parameters (GPUs, CPUs, memory, time limits), and submit deployment jobs directly from a user-friendly web UI. The vLLM service runs in the foreground of the submitted Slurm job, ensuring its lifecycle is managed by Slurm and its logs are captured in the standard Slurm output/error files.

The system consists of:
1.  A **Frontend**: An HTML page with JavaScript logic that captures user configurations.
2.  A **Backend**: A Python FastAPI server that receives configurations from the frontend, generates a `.slurm` script, and submits it to the Slurm scheduler using `sbatch`.
3.  **Installation & Management Scripts**: Shell scripts to install dependencies, set up the environment, and start/stop the frontend and backend servers.

## Key Features

* Web-based UI for easy vLLM deployment.
* Dynamic generation of Slurm batch scripts.
* Direct submission of jobs to Slurm via a backend API.
* Predefined list of popular LLMs.
* Configurable Slurm parameters (job name, resources, email notifications).
* Conda environment management for dependencies.
* Separate start/stop scripts for managing the SERAPHIM application servers.
* vLLM service runs in the foreground of the Slurm job for better management and logging.
* Automatic `ulimit` adjustment within the Slurm script.

## Prerequisites

* A Linux environment with Slurm workload manager installed and configured.
* Conda (Miniconda or Anaconda) installed.
* Access to a user account that has permissions to submit jobs to Slurm (`sbatch` command must be usable by the user running the backend).
* NVIDIA GPUs accessible via Slurm for vLLM.
* Python 3.10 (the installation script will create a Conda environment with this version).
* Internet access for downloading dependencies and models.

## Installation

1.  **Clone/Download the Project:**
    Obtain all project files, including `install.sh`, `seraphim_deploy.html` (template), and other related files if they are separate. Ensure they are in a root directory for the project (e.g., `~/SERAPHIM_PROJECT`).

2.  **Run the Installation Script:**
    Navigate to the project's root directory in your terminal and execute the installation script:
    ```bash
    cd ~/SERAPHIM_PROJECT # Or your chosen project directory
    bash install.sh
    ```
    The `install.sh` script will perform the following actions:
    * Create necessary directories (e.g., `$HOME/SERAPHIM`, `$HOME/SERAPHIM/scripts`).
    * Create a `vllm_requirements.txt` file.
    * Set up a Conda environment named `seraphim_vllm_env` with Python 3.10.
    * Install all required Python dependencies, including PyTorch, vLLM, FastAPI, and Uvicorn.
    * Generate the frontend (`seraphim_deploy.html`, `seraphim_logic.js`) and backend (`seraphim_backend.py`) files, configuring them with appropriate paths and ports.
    * Generate helper scripts: `start_seraphim.sh` and `stop_seraphim.sh` in the `$HOME/SERAPHIM` directory.

## Running SERAPHIM

After successful installation, you can start and stop the SERAPHIM application using the generated scripts.

1.  **Navigate to the SERAPHIM Directory:**
    ```bash
    cd $HOME/SERAPHIM
    ```

2.  **Start the Application:**
    ```bash
    ./start_seraphim.sh
    ```
    This script will:
    * Activate the `seraphim_vllm_env` Conda environment.
    * Start the Python FastAPI backend server in the background (default port: 8870). Logs are saved to `seraphim_backend.log`.
    * Start a simple Python HTTP server to serve the frontend in the background (default port: 8869). Logs are saved to `seraphim_frontend.log`.
    * Display the URLs to access the frontend and monitor logs.

3.  **Access the Web UI:**
    Open your web browser and navigate to the frontend URL provided by the `start_seraphim.sh` script (e.g., `http://YOUR_SERVER_IP:8869`).

4.  **Stop the Application:**
    ```bash
    ./stop_seraphim.sh
    ```
    This script will stop the background backend and frontend server processes.

## Using the Web Interface

1.  **Open the SERAPHIM UI** in your browser.
2.  **Select a Model:** Choose a vLLM model from the dropdown list.
3.  **Configure Service Port:** Specify the port on which the vLLM service will listen on the allocated Slurm node (default is 8000).
4.  **Hugging Face Token (Optional):** If the selected model is gated or requires authentication, enter your Hugging Face token.
5.  **Slurm Configuration:**
    * **Job Name:** A descriptive name for your Slurm job.
    * **Time Limit:** Maximum run time for the job (e.g., `23:59:59`).
    * **GPUs:** Number and type of GPUs (e.g., `1`, `a100:2`).
    * **CPUs per Task:** Number of CPU cores.
    * **Memory:** Memory allocation (e.g., `32G`).
    * **Email Notify (Optional):** Your email address for Slurm job notifications.
6.  **Deploy:** Click the "Deploy to Slurm via Backend" button.
7.  **Monitor:**
    * The UI will display a confirmation message from the backend, including the submitted Slurm Job ID and paths to the Slurm output/error files.
    * The vLLM service logs (including the API endpoint URL once the service starts) will be written to the Slurm output file (e.g., `$HOME/SERAPHIM/scripts/your_job_name_JOBID.out`).
    * Check these files on the server to monitor the job's progress and find the vLLM API endpoint.

## Directory Structure (within `$HOME/SERAPHIM`)

* `seraphim_deploy.html`: The main HTML file for the frontend.
* `seraphim_logic.js`: JavaScript logic for the frontend.
* `seraphim_backend.py`: Python FastAPI backend server script.
* `start_seraphim.sh`: Script to start the frontend and backend servers.
* `stop_seraphim.sh`: Script to stop the frontend and backend servers.
* `seraphim_backend.log`: Log file for the backend server.
* `seraphim_frontend.log`: Log file for the frontend HTTP server.
* `seraphim_backend.pid`: PID file for the backend server process.
* `seraphim_frontend.pid`: PID file for the frontend server process.
* `scripts/`:
    * `vllm_requirements.txt`: Python dependencies.
    * `deploy_*.slurm`: Dynamically generated Slurm scripts are saved here by the backend.
    * `*.out`, `*.err`: Output and error files from Slurm jobs will be placed here (as configured in the Slurm scripts).
    * `vllm_service_specific_logs/`: This directory was initially planned for separate vLLM logs but is less critical now as vLLM output goes to main Slurm .out/.err files. It might be used by the backend for other purposes in the future.

## Important Notes

* **Permissions:** The user running the `seraphim_backend.py` script (via `start_seraphim.sh`) must have `sbatch` command available in their PATH and the necessary permissions to submit jobs to the Slurm cluster.
* **CORS:** The backend server (`seraphim_backend.py`) is configured with CORS to allow requests from any origin (`allow_origins=["*"]`) for ease of development. For a production environment, you should restrict this to the specific origin of your frontend.
* **Server Management:** The `start_seraphim.sh` script runs the servers in the background using `nohup`. For robust production deployment, consider using a process manager like `systemd`, `supervisor`, or `tmux`/`screen`.
* **Model Compatibility:** The `max-model-len` is adjusted for some models (Llama-2, Mixtral) in the backend. Ensure these values are appropriate for your models and available VRAM. You might need to adjust this logic in `seraphim_backend.py` if you add more models or encounter issues.
* **Error Handling:** The scripts include basic error handling. Always check the server logs (`seraphim_backend.log`, `seraphim_frontend.log`) and Slurm output/error files for troubleshooting.

## Future Features & Enhancements

* **Dynamic Endpoint Listing:** Implement functionality in the "Active Deployments" section of the UI to list currently running Slurm jobs submitted by SERAPHIM. This would require the backend to:
    * Query Slurm (`squeue`).
    * Parse Slurm output or vLLM log files to find active endpoints (node, port).
* **Job Management:** Add features to cancel or view detailed logs of submitted jobs directly from the UI.
* **User Authentication:** Implement user authentication for accessing SERAPHIM, especially if deployed in a shared environment.
* **Custom Model Paths:** Allow users to specify paths to custom models on the server instead of only Hugging Face model IDs.
* **Advanced vLLM Parameters:** Expose more vLLM serving parameters in the UI (e.g., `gpu_memory_utilization`, `dtype`, `tensor_parallel_size`).
* **Input Validation:** Enhance input validation on both frontend and backend for all parameters.
* **Configuration File:** Move hardcoded settings (ports, default Slurm values) to a configuration file.
* **Log Streaming:** Stream Slurm job logs directly to the web UI.
* **Scalability:** For many users, consider a more robust backend setup (e.g., Gunicorn with Uvicorn workers).
* **Security Hardening:** Thoroughly review and harden security aspects for production deployment (input sanitization, rate limiting, etc.).
* **Theme Customization:** Allow users to select different UI themes.
* **Dockerization:** Provide Dockerfiles for easier deployment of the SERAPHIM application stack.
* **Persistent Job Database:** Store information about submitted jobs in a database for better tracking and management.

