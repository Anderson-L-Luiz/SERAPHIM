#!/bin/bash

# SERAPHIM Installation Script - models.txt, Searchable Dropdown, Port Checks, Active Deployments
# vLLM service runs in FOREGROUND within Slurm job.

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
CONDA_ENV_NAME="seraphim_vllm_env"
SERAPHIM_DIR="$HOME/SERAPHIM"
SCRIPTS_DIR="$SERAPHIM_DIR/scripts" 
VLLM_LOG_DIR_IN_INSTALLER="$SERAPHIM_DIR/seraphim_internal_logs" 
HTML_FILENAME="seraphim_deploy.html"
JS_FILENAME="seraphim_logic.js"
BACKEND_FILENAME="seraphim_backend.py"
MODELS_FILENAME="models.txt" # New models file
START_SCRIPT_FILENAME="start_seraphim.sh"
STOP_SCRIPT_FILENAME="stop_seraphim.sh"

HTML_TARGET_PATH="$SERAPHIM_DIR/$HTML_FILENAME"
JS_TARGET_PATH="$SERAPHIM_DIR/$JS_FILENAME"
BACKEND_TARGET_PATH="$SERAPHIM_DIR/$BACKEND_FILENAME"
MODELS_FILE_PATH="$SERAPHIM_DIR/$MODELS_FILENAME" # Path for models.txt
START_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$START_SCRIPT_FILENAME"
STOP_SCRIPT_TARGET_PATH="$SERAPHIM_DIR/$STOP_SCRIPT_FILENAME"
VLLM_REQUIREMENTS_FILE="$SCRIPTS_DIR/vllm_requirements.txt"

BACKEND_PORT=8870
FRONTEND_PORT=8869
JOB_NAME_PREFIX_FOR_SQ="vllm_service" 

if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda command not found."
    exit 1
fi

echo "Starting SERAPHIM vLLM Deployment Setup..."
echo "Target Directory: $SERAPHIM_DIR"
echo "=========================================================================="
mkdir -p "$SERAPHIM_DIR"
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$VLLM_LOG_DIR_IN_INSTALLER" 
echo "Directories checked/created."
echo ""

# --- Create models.txt File ---
echo "Creating models list file: $MODELS_FILE_PATH"
cat > "$MODELS_FILE_PATH" << EOF_MODELS
# Format: model_id,Display Name
# --- PASTE YOUR FULL LIST OF MODELS BELOW ---
# Example models:
mistralai/Mistral-7B-Instruct-v0.1,Mistral-7B-Instruct-v0.1
meta-llama/Llama-2-7b-chat-hf,Llama-2-7B-Chat-HF (Meta)
google/gemma-1.1-7b-it,Gemma-1.1-7B-IT (Google)
Qwen/Qwen2-7B-Instruct,Qwen2-7B-Instruct (Alibaba)
BAAI/AquilaChat-7B,AquilaChat-7B (BAAI)
mistralai/Mixtral-8x7B-Instruct-v0.1,Mixtral-8x7B-Instruct-v0.1
EleutherAI/gpt-j-6b,GPT-J-6B (EleutherAI)
tiiuae/falcon-7b-instruct,Falcon-7B-Instruct (TII UAE)
mistralai/Pixtral-12B-2409,Pixtral-12B-2409 (MistralAI, Multimodal)
microsoft/phi-2,Phi-2 (Microsoft)
meta-llama/Meta-Llama-3-8B-Instruct,Llama-3-8B-Instruct (Meta)
meta-llama/Meta-Llama-3-70B-Instruct,Llama-3-70B-Instruct (Meta)
meta-llama/Meta-Llama-3.1-8B-Instruct,Llama-3.1-8B-Instruct (Meta)
meta-llama/Meta-Llama-3.1-70B-Instruct,Llama-3.1-70B-Instruct (Meta)
meta-llama/Meta-Llama-3.1-405B-Instruct,Llama-3.1-405B-Instruct (Meta)
google/gemma-1.1-2b-it,Gemma-1.1-2B-IT (Google)
google/gemma-2-9b-it,Gemma2-9B-IT (Google)
google/gemma-2-27b-it,Gemma2-27B-IT (Google)
Qwen/Qwen1.5-7B-Chat,Qwen1.5-7B-Chat (Alibaba)
Qwen/Qwen1.5-14B-Chat,Qwen1.5-14B-Chat (Alibaba)
Qwen/Qwen1.5-72B-Chat,Qwen1.5-72B-Chat (Alibaba)
Qwen/Qwen1.5-MoE-A2.7B-Chat,Qwen1.5-MoE-A2.7B-Chat (Alibaba)
Qwen/Qwen2-1.5B-Instruct,Qwen2-1.5B-Instruct (Alibaba)
Qwen/Qwen2-72B-Instruct,Qwen2-72B-Instruct (Alibaba)
microsoft/phi-3-mini-4k-instruct,Phi-3-Mini-4K-Instruct (Microsoft)
microsoft/phi-3-small-8k-instruct,Phi-3-Small-8K-Instruct (Microsoft)
microsoft/phi-3-medium-4k-instruct,Phi-3-Medium-4K-Instruct (Microsoft)
CohereForAI/c4ai-command-r-v01,Command-R v01 (CohereForAI)
CohereForAI/c4ai-command-r-plus,Command R+ (CohereForAI)
databricks/dbrx-instruct,DBRX Instruct (Databricks)
01-ai/Yi-6B-Chat,Yi-6B-Chat (01.AI)
01-ai/Yi-34B-Chat,Yi-34B-Chat (01.AI)
deepseek-ai/deepseek-llm-7b-chat,DeepSeek-LLM-7B-Chat (deepseek-ai)
deepseek-ai/deepseek-llm-67b-chat,DeepSeek-LLM-67B-Chat (deepseek-ai)
deepseek-ai/DeepSeek-V2-Chat,DeepSeek-V2-Chat (deepseek-ai)
deepseek-ai/deepseek-r1-7b-chat,DeepSeek-R1-7B-Chat (deepseek-ai)
deepseek-ai/deepseek-math-7b,DeepSeek-Math-7B (deepseek-ai)
deepseek-ai/deepseek-moe-16b-chat,DeepSeek-MoE-16B-Chat (deepseek-ai)
deepseek-ai/deepseek-coder-33b-instruct,DeepSeek-Coder-33B-Instruct (deepseek-ai)
togethercomputer/RedPajama-INCITE-Instruct-7B,RedPajama-INCITE-Instruct-7B (TogetherComputer)
mistralai/Mixtral-8x7B-Instruct,Mixtral-8x7B-Instruct (MistralAI)
tiiuae/falcon-40b-instruct,Falcon-40B-Instruct (TII UAE)
tiiuae/falcon-180B-chat,Falcon-180B-Chat (TII UAE)
EleutherAI/gpt-neox-20b,GPT-NeoX-20B (EleutherAI)
databricks/dolly-v2-12b,Dolly-v2-12B (Databricks)
bigscience/bloom,BLOOM (BigScience)
THUDM/chatglm3-6b,ChatGLM3-6B (THUDM)
internlm/internlm2-chat-7b,InternLM2-Chat-7B (InternLM)
internlm/internlm2-chat-20b,InternLM2-Chat-20B (InternLM)
mosaicml/mpt-7b-instruct,MPT-7B-Instruct (MosaicML)
mosaicml/mpt-30b-instruct,MPT-30B-Instruct (MosaicML)
WizardLM/WizardCoder-Python-34B-V1.0,WizardCoder-Python-34B (WizardLM)
llava-hf/llava-1.5-7b-hf,LLaVA-1.5-7B-HF (Multimodal)
llava-hf/llava-v1.6-mistral-7b-hf,LLaVA-1.6-Mistral-7B-HF (Multimodal)
microsoft/Phi-3-vision-128k-instruct,Phi-3-Vision-128K-Instruct (Microsoft, Multimodal)
Qwen/Qwen-VL-Chat,Qwen-VL-Chat (Alibaba, Multimodal)
google/paligemma-3b-mix-448,PaliGemma-3B-Mix-448 (Google, Multimodal)
BAAI/Aquila-7B,Aquila-7B (BAAI)
Snowflake/snowflake-arctic-base,Snowflake Arctic Base
Snowflake/snowflake-arctic-instruct,Snowflake Arctic Instruct
baichuan-inc/Baichuan2-7B-Chat,Baichuan2-7B-Chat (baichuan-inc)
baichuan-inc/Baichuan2-13B-Chat,Baichuan2-13B-Chat (baichuan-inc)
baichuan-inc/Baichuan-7B,Baichuan-7B (baichuan-inc)
baichuan-inc/Baichuan-13B-Chat,Baichuan-13B-Chat (baichuan-inc)
bigscience/bloom-560m,BLOOM-560m (BigScience)
bigscience/bloomz-560m,BLOOMz-560m (BigScience)
THUDM/chatglm2-6b,ChatGLM2-6B (THUDM)
CohereForAI/c4ai-command-r7b-12-2024,Command-R 7B (CohereForAI, 12-2024)
databricks/dbrx-base,DBRX Base (Databricks)
Deci/DeciLM-7B,DeciLM-7B (Deci)
Deci/DeciLM-7B-instruct,DeciLM-7B-Instruct (Deci)
deepseek-ai/deepseek-llm-7b-base,DeepSeek-LLM-7B-Base (deepseek-ai)
deepseek-ai/deepseek-llm-67b-base,DeepSeek-LLM-67B-Base (deepseek-ai)
deepseek-ai/DeepSeek-V2-Base,DeepSeek-V2-Base (deepseek-ai)
deepseek-ai/DeepSeek-V3-Base,DeepSeek-V3-Base (deepseek-ai)
deepseek-ai/DeepSeek-V3-Chat,DeepSeek-V3-Chat (deepseek-ai)
LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct,EXAONE-3.0-7.8B-Instruct (LGAI)
tiiuae/falcon-7b,Falcon-7B (TII UAE)
tiiuae/falcon-40b,Falcon-40B (TII UAE)
tiiuae/falcon-rw-7b,Falcon-RW-7B (TII UAE)
tiiuae/falcon-mamba-7b,FalconMamba-7B (TII UAE)
tiiuae/falcon-mamba-7b-instruct,FalconMamba-7B-Instruct (TII UAE)
google/gemma-2b,Gemma-2B (Google)
google/gemma-7b,Gemma-7B (Google)
google/gemma-2-9b,Gemma2-9B (Google)
google/gemma-2-27b,Gemma2-27B (Google)
google/gemma-3-1b-it,Gemma3-1B-IT (Google)
THUDM/glm-4-9b-chat-hf,GLM-4-9B-Chat (THUDM)
gpt2,GPT-2 (OpenAI)
gpt2-medium,GPT-2-Medium (OpenAI)
gpt2-large,GPT-2-Large (OpenAI)
gpt2-xl,GPT-2-XL (OpenAI)
bigcode/starcoder,StarCoder (BigCode)
bigcode/starcoderbase,StarCoderBase (BigCode)
bigcode/gpt_bigcode-santacoder,SantaCoder (BigCode)
WizardLM/WizardCoder-15B-V1.0,WizardCoder-15B (WizardLM)
nomic-ai/gpt4all-j,GPT4All-J (NomicAI)
EleutherAI/pythia-6.9b,Pythia-6.9B (EleutherAI)
EleutherAI/pythia-12b,Pythia-12B (EleutherAI)
OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5,OpenAssistant Pythia-12B (OASST)
stabilityai/stablelm-base-alpha-7b-v2,StableLM Base Alpha 7B v2 (StabilityAI)
stabilityai/stablelm-tuned-alpha-7b,StableLM Tuned Alpha 7B (StabilityAI)
ibm-granite/granite-3.0-2b-base,Granite-3.0-2B-Base (IBM)
ibm-granite/granite-3.1-8b-instruct,Granite-3.1-8B-Instruct (IBM)
ibm/PowerLM-3b,PowerLM-3B (IBM)
ibm-granite/granite-3.0-1b-a400m-base,Granite-3.0-1B-MoE-Base (IBM)
ibm-granite/granite-3.0-3b-a800m-instruct,Granite-3.0-3B-MoE-Instruct (IBM)
parasail-ai/GritLM-7B-vllm,GritLM-7B (Parasail AI)
hpcai-tech/grok-1,Grok-1 (xAI/hpcai-tech)
internlm/internlm-7b,InternLM-7B (InternLM)
internlm/internlm-chat-7b,InternLM-Chat-7B (InternLM)
internlm/internlm2-7b,InternLM2-7B (InternLM)
internlm/internlm2-20b,InternLM2-20B (InternLM)
internlm/internlm3-8b-instruct,InternLM3-8B-Instruct (InternLM)
inceptionai/jais-13b,Jais-13B (InceptionAI)
inceptionai/jais-13b-chat,Jais-13B-Chat (InceptionAI)
inceptionai/jais-30b-v3,Jais-30B-v3 (InceptionAI)
inceptionai/jais-30b-chat-v3,Jais-30B-Chat-v3 (InceptionAI)
ai21labs/AI21-Jamba-1.5-Large,Jamba-1.5-Large (AI21 Labs)
ai21labs/AI21-Jamba-1.5-Mini,Jamba-1.5-Mini (AI21 Labs)
ai21labs/Jamba-v0.1,Jamba-v0.1 (AI21 Labs)
meta-llama/Llama-2-7b-hf,Llama-2-7B-HF (Meta)
meta-llama/Llama-2-13b-hf,Llama-2-13B-HF (Meta)
meta-llama/Llama-2-70b-hf,Llama-2-70B-HF (Meta)
meta-llama/Llama-2-13b-chat-hf,Llama-2-13B-Chat-HF (Meta)
meta-llama/Llama-2-70b-chat-hf,Llama-2-70B-Chat-HF (Meta)
meta-llama/Meta-Llama-3-8B,Llama-3-8B (Meta)
meta-llama/Meta-Llama-3-70B,Llama-3-70B (Meta)
meta-llama/Meta-Llama-3.1-70B,Llama-3.1-70B (Meta)
01-ai/Yi-6B,Yi-6B (01.AI)
01-ai/Yi-34B,Yi-34B (01.AI)
state-spaces/mamba-130m-hf,Mamba-130M (state-spaces)
state-spaces/mamba-790m-hf,Mamba-790M (state-spaces)
state-spaces/mamba-2.8b-hf,Mamba-2.8B (state-spaces)
openbmb/MiniCPM-2B-sft-bf16,MiniCPM-2B-SFT (OpenBMB)
openbmb/MiniCPM-2B-dpo-bf16,MiniCPM-2B-DPO (OpenBMB)
mistralai/Mistral-7B-v0.1,Mistral-7B-v0.1
mistralai/Mixtral-8x7B-v0.1,Mixtral-8x7B-v0.1
mistralai/Mixtral-8x22B-v0.1,Mixtral-8x22B-v0.1
mistralai/Mixtral-8x22B-Instruct-v0.1,Mixtral-8x22B-Instruct-v0.1
mosaicml/mpt-7b,MPT-7B (MosaicML)
mosaicml/mpt-7b-chat,MPT-7B-Chat (MosaicML)
mosaicml/mpt-30b,MPT-30B (MosaicML)
mosaicml/mpt-30b-chat,MPT-30B-Chat (MosaicML)
allenai/OLMo-1B,OLMo-1B (AllenAI)
allenai/OLMo-7B,OLMo-7B (AllenAI)
allenai/OLMo-7B-Instruct,OLMo-7B-Instruct (AllenAI)
facebook/opt-125m,OPT-125m (Facebook)
facebook/opt-350m,OPT-350m (Facebook)
facebook/opt-1.3b,OPT-1.3B (Facebook)
facebook/opt-2.7b,OPT-2.7B (Facebook)
facebook/opt-6.7b,OPT-6.7B (Facebook)
facebook/opt-13b,OPT-13B (Facebook)
facebook/opt-30b,OPT-30B (Facebook)
facebook/opt-66b,OPT-66B (Facebook)
OrionStarAI/Orion-14B-Base,Orion-14B-Base (OrionStarAI)
OrionStarAI/Orion-14B-Chat,Orion-14B-Chat (OrionStarAI)
microsoft/phi-1,Phi-1 (Microsoft)
microsoft/phi-1_5,Phi-1.5 (Microsoft)
Qwen/Qwen-1_8B,Qwen-1.8B (Alibaba)
Qwen/Qwen-7B,Qwen-7B (Alibaba)
Qwen/Qwen-14B,Qwen-14B (Alibaba)
Qwen/Qwen-72B,Qwen-72B (Alibaba)
Qwen/Qwen-1_8B-Chat,Qwen-1.8B-Chat (Alibaba)
Qwen/Qwen-7B-Chat,Qwen-7B-Chat (Alibaba)
Qwen/Qwen-14B-Chat,Qwen-14B-Chat (Alibaba)
Qwen/Qwen-72B-Chat,Qwen-72B-Chat (Alibaba)
Qwen/Qwen1.5-MoE-A2.7B,Qwen1.5-MoE-A2.7B (Alibaba)
stabilityai/stablelm-3b-4e1t,StableLM-3B-4E1T (StabilityAI)
stabilityai/stablelm-2-zephyr-1_6b,StableLM-2-Zephyr-1.6B (StabilityAI)
bigcode/starcoder2-3b,StarCoder2-3B (BigCode)
bigcode/starcoder2-7b,StarCoder2-7B (BigCode)
bigcode/starcoder2-15b,StarCoder2-15B (BigCode)
xverse/XVERSE-7B,XVERSE-7B (xverse)
xverse/XVERSE-13B,XVERSE-13B (xverse)
xverse/XVERSE-65B,XVERSE-65B (xverse)
xverse/XVERSE-7B-Chat,XVERSE-7B-Chat (xverse)
xverse/XVERSE-13B-Chat,XVERSE-13B-Chat (xverse)
xverse/XVERSE-65B-Chat,XVERSE-65B-Chat (xverse)
Salesforce/blip2-opt-2.7b,BLIP-2-OPT-2.7B (Salesforce, Multimodal)
Salesforce/blip2-opt-6.7b,BLIP-2-OPT-6.7B (Salesforce, Multimodal)
Salesforce/blip2-flan-t5-xl,BLIP-2-Flan-T5-XL (Salesforce, Multimodal)
facebook/chameleon-7b,Chameleon-7B (Facebook, Multimodal)
adept/fuyu-8b,Fuyu-8B (Adept, Multimodal)
HuggingFaceM4/idefics2-8b,Idefics2-8B (HuggingFaceM4, Multimodal)
HuggingFaceM4/idefics2-8b-chatty,Idefics2-8B-Chatty (HuggingFaceM4, Multimodal)
OpenGVLab/InternVL2-4B,InternVL2-4B (OpenGVLab, Multimodal)
OpenGVLab/InternVL2-8B,InternVL2-8B (OpenGVLab, Multimodal)
OpenGVLab/InternVL2-26B,InternVL2-26B (OpenGVLab, Multimodal)
llava-hf/llava-1.5-13b-hf,LLaVA-1.5-13B-HF (Multimodal)
llava-hf/llava-v1.6-vicuna-7b-hf,LLaVA-1.6-Vicuna-7B-HF (Multimodal)
llava-hf/llava-v1.6-vicuna-13b-hf,LLaVA-1.6-Vicuna-13B-HF (Multimodal)
llava-hf/llava-v1.6-34b-hf,LLaVA-1.6-34B-HF (Multimodal)
openbmb/MiniCPM-V-2,MiniCPM-V-2 (OpenBMB, Multimodal)
openbmb/MiniCPM-Llama3-V-2_5,MiniCPM-Llama3-V-2.5 (OpenBMB, Multimodal)
google/paligemma-3b-pt-224,PaliGemma-3B-PT-224 (Google, Multimodal)
google/paligemma-3b-mix-224,PaliGemma-3B-Mix-224 (Google, Multimodal)
microsoft/Phi-3.5-vision-instruct,Phi-3.5-Vision-Instruct (Microsoft, Multimodal)
Qwen/Qwen2-VL-2B-Instruct,Qwen2-VL-2B-Instruct (Alibaba, Multimodal)
Qwen/Qwen2-VL-7B-Instruct,Qwen2-VL-7B-Instruct (Alibaba, Multimodal)
Qwen/Qwen2-VL-72B-Instruct,Qwen2-VL-72B-Instruct (Alibaba, Multimodal)
EOF_MODELS
echo "Models file created with examples. Please populate it with your full list."
echo ""

echo "Creating requirements file: $VLLM_REQUIREMENTS_FILE"
cat > "$VLLM_REQUIREMENTS_FILE" << EOF
# Core vLLM and serving
vllm>=0.4.0
# Backend requirements
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
python-dotenv>=1.0.0
# Other vLLM dependencies
aiohappyeyeballs>=2.4.0
aiohttp>=3.8.0
aiohttp-cors>=0.7.0
huggingface-hub>=0.20.0
numpy>=1.23.0
openai>=1.0.0
packaging>=23.0
prometheus-client>=0.17.0
protobuf>=4.20.0
pydantic_core>=2.0.0
PyYAML>=6.0.0
ray>=2.5.0
requests>=2.30.0
safetensors>=0.4.0
sentencepiece>=0.1.98
tokenizers>=0.14.0
torch>=2.1.0
torchaudio>=2.1.0
torchvision>=0.16.0
transformers>=4.35.0
typing_extensions>=4.8.0
xformers>=0.0.22
EOF
echo "Requirements file created."
echo ""

echo "Setting up Conda environment: $CONDA_ENV_NAME"
if conda env list | grep -q "$CONDA_ENV_NAME"; then
    conda env remove -n "$CONDA_ENV_NAME" -y
fi
conda create -n "$CONDA_ENV_NAME" python=3.10 -y

echo "Sourcing conda for activation (during install)..."
CONDA_BASE_PATH=$(conda info --base)
if [ -z "$CONDA_BASE_PATH" ]; then
    echo "❌ Error: Could not determine Conda base path."
    exit 1
fi
echo "Detected CONDA_BASE_PATH: $CONDA_BASE_PATH"
CONDA_SH_PATH="$CONDA_BASE_PATH/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH_PATH" ]; then
    echo "❌ Error: conda.sh not found at $CONDA_SH_PATH."
    exit 1
fi
# shellcheck source=/dev/null
. "$CONDA_SH_PATH"
conda activate "$CONDA_ENV_NAME"
if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    echo "❌ Error: Failed to activate conda environment '$CONDA_ENV_NAME'."
    exit 1
fi
echo "✅ Conda environment '$CONDA_ENV_NAME' activated."
echo ""

echo "Installing Python dependencies into '$CONDA_ENV_NAME'..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then echo "❌ Error: Failed PyTorch install."; exit 1; fi
python -m pip install -r "$VLLM_REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then echo "❌ Error: Failed main requirements install."; exit 1; fi
echo "✅ All Python dependencies installed."
echo ""

echo "Generating Backend Python script: $BACKEND_TARGET_PATH"
cat > "$BACKEND_TARGET_PATH" << 'PYTHON_EOF'
# seraphim_backend.py

import os
import subprocess
import uuid
import re # For parsing squeue and log files
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime
import logging
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

SERAPHIM_DIR_PY = "{{SERAPHIM_DIR_PLACEHOLDER}}"
SCRIPTS_DIR_PY = "{{SCRIPTS_DIR_PLACEHOLDER}}" 
VLLM_LOG_DIR_PY = "{{VLLM_LOG_DIR_PLACEHOLDER}}" 
CONDA_ENV_NAME_PY = "{{CONDA_ENV_NAME_PLACEHOLDER}}"
BACKEND_PORT_PY = {{BACKEND_PORT_PLACEHOLDER}}
# Prefix used to identify SERAPHIM jobs in squeue, ensure it matches job names from UI
JOB_NAME_PREFIX_FOR_SQ_PY = "{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}" 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["POST", "GET"], allow_headers=["*"],
)

class SlurmConfig(BaseModel):
    selected_model: str; service_port: str; hf_token: str | None = None
    job_name: str; time_limit: str; gpus: str; cpus_per_task: str; mem: str
    mail_user: str | None = None

class DeployedServiceInfo(BaseModel):
    job_id: str
    job_name: str
    status: str
    nodes: Optional[str] = None
    partition: Optional[str] = None
    time_used: Optional[str] = None
    user: Optional[str] = None
    service_url: Optional[str] = None
    slurm_output_file: Optional[str] = None
    raw_squeue_line: Optional[str] = None


def generate_sbatch_script_content(config: SlurmConfig, scripts_dir: str, conda_env_name: str) -> tuple[str, str, str, str]:
    conda_base_path_for_slurm_script = "$(conda info --base)"
    escaped_selected_model_for_vllm_cmd = config.selected_model.replace('"', '\\"')

    vllm_serve_command = f'vllm serve "{escaped_selected_model_for_vllm_cmd}"'
    model_args = [
        f'--host "0.0.0.0"', f'--port {config.service_port}', '--trust-remote-code'
    ]
    max_model_len = 16384
    if "llama-2-7b" in config.selected_model.lower() or "llama2-7b" in config.selected_model.lower():
        max_model_len = 4096 
        logger.info(f"Adjusted max_model_len to {max_model_len} for {config.selected_model}")
    elif "mixtral" in config.selected_model.lower():
        max_model_len = 32768
        logger.info(f"Adjusted max_model_len to {max_model_len} for {config.selected_model}")
    
    if "pixtral" in config.selected_model.lower():
        model_args.append('--guided-decoding-backend=lm-format-enforcer')
        model_args.append("--limit_mm_per_prompt 'image=8'")
        if "mistralai/Pixtral-12B-2409" in config.selected_model:
             model_args.extend(['--enable-auto-tool-choice', '--tool-call-parser=mistral',
                                '--tokenizer_mode mistral', '--revision aaef4baf771761a81ba89465a18e4427f3a105f9'])
    model_args.append(f'--max-model-len {max_model_len}')
    vllm_serve_command_full = vllm_serve_command + " \\\n    " + " \\\n    ".join(model_args)

    mail_type_line = f"#SBATCH --mail-type=ALL\\n#SBATCH --mail-user={config.mail_user}" if config.mail_user else "#SBATCH --mail-type=NONE"
    safe_filename_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config.job_name)
    unique_id = str(uuid.uuid4())[:8]
    script_filename = f"deploy_{safe_filename_job_name}_{unique_id}.slurm" # Ensure this job_name part matches what squeue filters on if needed
    script_path = os.path.join(scripts_dir, script_filename)

    slurm_out_file = os.path.join(scripts_dir, f"{safe_filename_job_name}_%j.out")
    slurm_err_file = os.path.join(scripts_dir, f"{safe_filename_job_name}_%j.err")

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={config.job_name} 
#SBATCH --output={slurm_out_file}
#SBATCH --error={slurm_err_file}
#SBATCH --time={config.time_limit}
#SBATCH --gres=gpu:{config.gpus}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.mem}
{mail_type_line}

echo "Current ulimit -n (soft): $(ulimit -Sn)"
echo "Current ulimit -n (hard): $(ulimit -Hn)"
ulimit -n 10240 
if [ $? -eq 0 ]; then echo "Successfully set ulimit -n to $(ulimit -Sn)"; else echo "WARN: Failed to set ulimit -n. Current: $(ulimit -Sn)."; fi
echo "=================================================================="
echo "✝ SERAPHIM vLLM Deployment Job - SLURM PREP ✝"
echo "Job Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID running on Node: $(hostname -f) (Short: $(hostname -s))"
echo "Slurm Output File: {slurm_out_file.replace('%j', '$SLURM_JOB_ID')}"
echo "Slurm Error File: {slurm_err_file.replace('%j', '$SLURM_JOB_ID')}"
echo "Model: {config.selected_model}"
echo "Target Service Port: {config.service_port}"
echo "Conda Env: {conda_env_name}"
echo "Max Model Length: {max_model_len}"
echo "vLLM service will run in the FOREGROUND of this Slurm job."
echo "=================================================================="

CONDA_BASE_PATH_SLURM="{conda_base_path_for_slurm_script}"
if [ -z "$CONDA_BASE_PATH_SLURM" ]; then echo "ERROR: Conda base path empty."; exit 1; fi
CONDA_SH_PATH="$CONDA_BASE_PATH_SLURM/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH_PATH" ]; then . "$CONDA_SH_PATH"; else echo "WARN: conda.sh not found."; fi

conda activate "{conda_env_name}"
if [[ "$CONDA_PREFIX" != *"{conda_env_name}"* ]]; then echo "ERROR: Failed to activate conda. Prefix: $CONDA_PREFIX"; exit 1; fi
echo "Conda env '{conda_env_name}' activated. Path: $CONDA_PREFIX";

HF_TOKEN_VALUE="{config.hf_token or ''}"
if [ -n "$HF_TOKEN_VALUE" ]; then export HF_TOKEN="$HF_TOKEN_VALUE"; echo "HF_TOKEN set."; else echo "HF_TOKEN not provided."; fi

export VLLM_CONFIGURE_LOGGING="0"; export VLLM_NO_USAGE_STATS="True"; export VLLM_DO_NOT_TRACK="True"
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1" 

echo -e "\\nStarting vLLM API Server in FOREGROUND..."
echo "Command: {vllm_serve_command_full}"
echo "vLLM logs will be in Slurm output/error files."
echo "--- vLLM Service Starting (Output will follow) ---"

{vllm_serve_command_full}

VLLM_EXIT_CODE=$?
echo "--- vLLM Service Ended (Exit Code: $VLLM_EXIT_CODE) ---"
echo "=================================================================="
echo "✝ SERAPHIM vLLM Job - FINAL STATUS ✝"
if [ $VLLM_EXIT_CODE -eq 0 ]; then echo "vLLM exited cleanly or was terminated."; else echo "ERROR: vLLM exited with code: $VLLM_EXIT_CODE."; fi
echo "Slurm job $SLURM_JOB_ID finished."
echo "=================================================================="
"""
    return script_path, sbatch_content, slurm_out_file.replace('%j', '$SLURM_JOB_ID'), slurm_err_file.replace('%j', '$SLURM_JOB_ID')

@app.post("/api/deploy")
async def deploy_vllm_service_api(config: SlurmConfig, request: Request):
    logger.info(f"Deployment request for model: {config.selected_model}")
    try: os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    except OSError as e: raise HTTPException(status_code=500, detail=f"Server error creating script dir: {e}")

    script_path, sbatch_content, slurm_out_pattern, slurm_err_pattern = generate_sbatch_script_content(
        config, SCRIPTS_DIR_PY, CONDA_ENV_NAME_PY
    )
    try:
        with open(script_path, "w") as f: f.write(sbatch_content)
        os.chmod(script_path, 0o755)
        logger.info(f"Slurm script saved: {script_path}")
    except IOError as e: raise HTTPException(status_code=500, detail=f"Server error writing script: {e}")

    try:
        submit_command = ["sbatch", script_path]
        process = subprocess.run(submit_command, capture_output=True, text=True, check=True, timeout=30)
        job_id_message = process.stdout.strip()
        job_id = job_id_message.split(" ")[-1].strip() if "Submitted batch job" in job_id_message else "Unknown"
        logger.info(f"Sbatch successful. Output: '{job_id_message}', Parsed Job ID: {job_id}")
        
        actual_slurm_out = slurm_out_pattern.replace('$SLURM_JOB_ID', job_id if job_id != "Unknown" else "<JOB_ID>")
        actual_slurm_err = slurm_err_pattern.replace('$SLURM_JOB_ID', job_id if job_id != "Unknown" else "<JOB_ID>")

        return {"message": f"Slurm job submitted! ({job_id_message})", "job_id": job_id,
                "script_path": script_path, "slurm_output_file_pattern": actual_slurm_out,
                "slurm_error_file_pattern": actual_slurm_err,
                "monitoring_note": f"Monitor Slurm output ({actual_slurm_out}) for service logs and errors."}
    except subprocess.TimeoutExpired: raise HTTPException(status_code=500, detail="sbatch command timed out.")
    except subprocess.CalledProcessError as e:
        detail_msg = f"Sbatch failed. RC: {e.returncode}. Stderr: {e.stderr.strip()}" if e.stderr.strip() else "Sbatch failed."
        raise HTTPException(status_code=500, detail=detail_msg)
    except FileNotFoundError: raise HTTPException(status_code=500, detail="sbatch not found.")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Unexpected sbatch error: {str(e)}")

def parse_slurm_log_for_url(log_file_path: str, job_node: str, job_port: str) -> Optional[str]:
    """Tries to find the Uvicorn URL in the first N lines of a log file."""
    if not os.path.exists(log_file_path):
        logger.debug(f"Log file not found for URL parsing: {log_file_path}")
        return None
    try:
        with open(log_file_path, 'r', errors='ignore') as f: # Add errors='ignore' for robustness
            for _ in range(200): # Check first 200 lines
                line = f.readline()
                if not line: break
                match = re.search(r"Uvicorn running on http://([\d\.]+):(\d+)", line)
                if match:
                    log_ip, log_port_str = match.groups()
                    if log_port_str == job_port: 
                        service_host = job_node if job_node and log_ip == "0.0.0.0" else log_ip
                        if service_host:
                             return f"http://{service_host}:{job_port}/docs" 
            logger.debug(f"Uvicorn URL line not found in first 200 lines of {log_file_path}")
            if job_node and job_port: 
                return f"http://{job_node}:{job_port}/docs (best guess, check log)"
    except Exception as e:
        logger.warning(f"Could not read or parse log file {log_file_path}: {e}")
    return None

@app.get("/api/active_deployments", response_model=List[DeployedServiceInfo])
async def get_active_deployments():
    deployments = []
    try:
        user = subprocess.check_output("whoami", text=True).strip()
        squeue_cmd = ["squeue", "-u", user, "-o", "%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R", 
                      "--noheader", "--states=RUNNING,PENDING"]
        process = subprocess.run(squeue_cmd, capture_output=True, text=True, check=True, timeout=15)
        
        for line in process.stdout.strip().split('\n'):
            if not line.strip(): continue
            try:
                parts = line.split()
                job_id, partition = parts[0], parts[1]
                node_list_val, nodes_count_val, time_val, state_val, user_val = "", "", "", "", ""
                job_name = " ".join(parts[2:-5]) # Default assumption

                if len(parts) >= 7: # A more robust way to parse squeue fixed format by field order
                    node_list_val = parts[-1]
                    nodes_count_val = parts[-2]
                    time_val = parts[-3]
                    state_val = parts[-4]
                    user_val = parts[-5]
                    job_name_parts = parts[2:-5]
                    job_name = " ".join(job_name_parts).strip()


                if not job_name.startswith(JOB_NAME_PREFIX_FOR_SQ_PY):
                    continue

                service_info = DeployedServiceInfo(
                    job_id=job_id, job_name=job_name, status=state_val,
                    nodes=node_list_val if node_list_val and node_list_val != "(None)" else None,
                    partition=partition, time_used=time_val, user=user_val,
                    raw_squeue_line=line
                )

                if service_info.status == "R" and service_info.nodes:
                    # Construct path to Slurm output file
                    # Assumes job_name from squeue matches the file naming convention part
                    # This needs to align with how #SBATCH --job-name is set and how slurm_out_file is constructed
                    # The #SBATCH --job-name is config.job_name
                    # The slurm_out_file is based on safe_filename_job_name = "".join(...) from config.job_name
                    # So, we should use the original config.job_name for constructing the log file path.
                    # This is hard to get here, so we use the squeue job_name as a proxy.
                    safe_squeue_job_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in job_name)
                    slurm_out_file_path = os.path.join(SCRIPTS_DIR_PY, f"{safe_squeue_job_name}_{job_id}.out")
                    service_info.slurm_output_file = slurm_out_file_path
                    
                    # Try to find service port. This is difficult without original config.
                    # For now, assume a default or try to parse from job name if encoded.
                    # This part is highly heuristic.
                    job_port_from_name = "8000" # Default
                    port_match_in_name = re.search(r"_p(\d{4,5})_", job_name) # e.g. my_job_p8001_model
                    if port_match_in_name:
                        job_port_from_name = port_match_in_name.group(1)
                    
                    service_info.service_url = parse_slurm_log_for_url(
                        slurm_out_file_path, 
                        service_info.nodes.split(',')[0], # Use first node if multiple
                        job_port_from_name 
                    )
                
                deployments.append(service_info)
            except Exception as e:
                logger.error(f"Error parsing squeue line '{line}': {e}", exc_info=False)
                deployments.append(DeployedServiceInfo(job_id="PARSE_ERROR", job_name=line, status="UNKNOWN"))
    except Exception as e:
        logger.error(f"Error fetching active deployments: {e}", exc_info=True)
    return deployments

if __name__ == "__main__":
    os.makedirs(SCRIPTS_DIR_PY, exist_ok=True)
    os.makedirs(VLLM_LOG_DIR_PY, exist_ok=True) 
    logger.info(f"Starting SERAPHIM Backend Server on http://0.0.0.0:{BACKEND_PORT_PY}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT_PY)
PYTHON_EOF
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write Backend Python script."; exit 1; fi;
ESCAPED_SERAPHIM_DIR_FOR_SED=$(printf '%s\n' "$SERAPHIM_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_SCRIPTS_DIR_FOR_SED=$(printf '%s\n' "$SCRIPTS_DIR" | sed 's:[&/\]:\\&:g')
ESCAPED_VLLM_LOG_DIR_FOR_SED=$(printf '%s\n' "$VLLM_LOG_DIR_IN_INSTALLER" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_ENV_NAME_FOR_SED=$(printf '%s\n' "$CONDA_ENV_NAME" | sed 's:[&/\]:\\&:g')
ESCAPED_CONDA_BASE_PATH_FOR_SED=$(printf '%s\n' "$CONDA_BASE_PATH" | sed 's:[&/\]:\\&:g')
ESCAPED_JOB_NAME_PREFIX_FOR_SQ_FOR_SED=$(printf '%s\n' "$JOB_NAME_PREFIX_FOR_SQ" | sed 's:[&/\]:\\&:g')

sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{SCRIPTS_DIR_PLACEHOLDER}}|$ESCAPED_SCRIPTS_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{VLLM_LOG_DIR_PLACEHOLDER}}|$ESCAPED_VLLM_LOG_DIR_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$BACKEND_TARGET_PATH"
sed -i "s|{{JOB_NAME_PREFIX_FOR_SQ_PLACEHOLDER}}|$ESCAPED_JOB_NAME_PREFIX_FOR_SQ_FOR_SED|g" "$BACKEND_TARGET_PATH"
echo "✅ Backend Python script ($BACKEND_FILENAME) configured."
echo ""

echo "Generating JavaScript logic file: $JS_TARGET_PATH";
cat > "$JS_TARGET_PATH" << 'EOF_JS'
// seraphim_logic.js
const BACKEND_API_URL_DEPLOY = `http://${window.location.hostname}:{{BACKEND_PORT_PLACEHOLDER}}/api/deploy`;
const BACKEND_API_URL_ACTIVE = `http://${window.location.hostname}:{{BACKEND_PORT_PLACEHOLDER}}/api/active_deployments`;
const MODELS_FILE_URL = 'models.txt'; // Assumes models.txt is in the same directory as HTML

let allModels = []; // To store all models fetched from models.txt

async function fetchAndPopulateModels() {
    console.log("SERAPHIM_DEBUG: Fetching models from", MODELS_FILE_URL);
    const modelSelect = document.getElementById('model-select');
    const modelSearchInput = document.getElementById('model-search');
    if (!modelSelect || !modelSearchInput) {
        console.error("SERAPHIM_DEBUG: Model select or search input not found!");
        return;
    }
    modelSelect.innerHTML = '<option value="">-- Loading models... --</option>'; // Clear previous options

    try {
        const response = await fetch(MODELS_FILE_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch models.txt: ${response.status} ${response.statusText}`);
        }
        const text = await response.text();
        const lines = text.split('\n');
        allModels = []; // Reset global models array
        lines.forEach(line => {
            line = line.trim();
            if (line && !line.startsWith('#')) { // Ignore empty lines and comments
                const parts = line.split(',');
                if (parts.length >= 2) {
                    const modelId = parts[0].trim();
                    const modelName = parts.slice(1).join(',').trim(); // Handle names with commas
                    allModels.push({ id: modelId, name: modelName });
                } else if (parts.length === 1 && parts[0]) { // Handle lines with only ID
                     allModels.push({ id: parts[0], name: parts[0] });
                }
            }
        });

        if (allModels.length === 0) {
            modelSelect.innerHTML = '<option value="">-- No models found in models.txt --</option>';
            console.warn("SERAPHIM_DEBUG: No models parsed from models.txt or file is empty/incorrectly formatted.");
            return;
        }
        
        // Sort models by name for better UX
        allModels.sort((a, b) => a.name.localeCompare(b.name));

        populateModelDropdown(allModels); // Initial population with all models
        console.log(`SERAPHIM_DEBUG: Successfully fetched and parsed ${allModels.length} models.`);

    } catch (error) {
        console.error("SERAPHIM_DEBUG: Error fetching or parsing models.txt:", error);
        modelSelect.innerHTML = '<option value="">-- Error loading models --</option>';
        const outputDiv = document.getElementById('output');
        if(outputDiv) outputDiv.textContent = `❌ Error loading models from models.txt: ${error.message}`;
    }
}

function populateModelDropdown(modelsToDisplay) {
    const modelSelect = document.getElementById('model-select');
    modelSelect.innerHTML = ''; // Clear existing options before populating

    const placeholderOption = document.createElement('option');
    placeholderOption.value = "";
    placeholderOption.textContent = "-- Select a Model --";
    modelSelect.appendChild(placeholderOption);

    modelsToDisplay.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.name; // Display name in dropdown
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
                    <strong>Node(s):</strong> ${job.nodes || 'N/A'}<br/>
                    <strong>User:</strong> ${job.user || 'N/A'}<br/>
                    <strong>Time Used:</strong> ${job.time_used || 'N/A'}<br/>
                    ${job.service_url ? `<strong>Service URL:</strong> <a href="${job.service_url}" target="_blank">${job.service_url}</a><br/>` : ''}
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
        outputDiv.textContent = `❌ Network/Connection Error: ${error.message}. Is backend at ${BACKEND_API_URL_DEPLOY} running?`;
    } finally {
        deployButton.disabled = false; deployButton.textContent = "Deploy to Slurm via Backend";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchAndPopulateModels(); // Load models from file first
    
    const modelSearchInput = document.getElementById('model-search');
    if (modelSearchInput) {
        modelSearchInput.addEventListener('input', filterModels);
    } else {
        console.error("SERAPHIM_DEBUG: Model search input not found!");
    }

    const deployBtn = document.getElementById('deploy-button');
    if (deployBtn) deployBtn.addEventListener('click', handleDeployClick);
    
    const refreshBtn = document.getElementById('refresh-endpoints-button');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshDeployedEndpoints);
    
    refreshDeployedEndpoints(); // Initial load of active deployments
});
EOF_JS
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write JavaScript file."; exit 1; fi;
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$JS_TARGET_PATH";
echo "✅ JavaScript logic file ($JS_FILENAME) configured."
echo ""

echo "Generating HTML file: $HTML_TARGET_PATH";
cat > "$HTML_TARGET_PATH" << 'EOF_HTML'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>✧ SERAPHIM CORE ✧ vLLM Deployment Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #029702; --secondary-color: #f3cb00; --accent-color: #ffcc00;
            --bg-color: #1c1c1e; --card-bg-color: #2c2c2e; --text-color: #e5e5e7;
            --text-muted-color: #8e8e93; --border-color: #3a3a3c;
            --font-body: 'Exo 2', sans-serif; --font-heading: 'Orbitron', sans-serif;
            --success-color: #02c702; --warning-color: #f3cb00; --error-color: #ff3b30;
        }
        body { font-family: var(--font-body); margin: 0; padding:0; background-color: var(--bg-color); color: var(--text-color); display: flex; flex-direction: column; min-height: 100vh; font-size: 16px; line-height: 1.6; }
        .header { background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%); color: white; padding: 20px 30px; text-align: center; border-bottom: 3px solid var(--accent-color); text-shadow: 0 1px 3px rgba(0,0,0,0.3); }
        .header h1 { margin: 0; font-family: var(--font-heading); font-size: 2.3em; font-weight: 700; letter-spacing: 2px; display: flex; align-items: center; justify-content: center; gap: 15px; }
        .header p { margin: 8px 0 0; font-size: 0.95em; opacity: 0.9; font-weight: 300;}
        .main-container { display: flex; flex-wrap: wrap; padding: 20px; gap: 20px; flex-grow: 1; max-width: 1300px; margin: 20px auto; width: 100%; box-sizing: border-box;}
        .form-container, .endpoints-container { background-color: var(--card-bg-color); padding: 25px; border-radius: 10px; border: 1px solid var(--border-color); box-sizing: border-box; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); }
        .form-container { flex: 2; min-width: 400px; }
        .endpoints-container { flex: 1; min-width: 380px; max-height: 80vh; overflow-y: auto; }
        h3 { font-family: var(--font-heading); color: var(--secondary-color); border-bottom: 1px solid var(--accent-color); padding-bottom: 10px; margin-top: 0; margin-bottom: 20px; font-size: 1.4em; letter-spacing: 1px; display: flex; align-items: center; gap: 10px; }
        label { display: block; margin-top: 15px; margin-bottom: 5px; font-weight: 600; font-size: 0.9em; color: var(--text-muted-color); text-transform: uppercase; letter-spacing: 0.5px;}
        #model-search { /* Style for the search input */
            width: calc(100% - 24px); /* Match other inputs */
            padding: 11px; margin-bottom: 8px; /* Spacing */
            border-radius: 5px; border: 1px solid var(--border-color);
            box-sizing: border-box; font-size: 0.95em;
            background-color: #3a3a3c; color: var(--text-color);
        }
        #model-search:focus {
             border-color: var(--primary-color);
             box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.3);
             outline: none; background-color: #4a4a4e;
        }
        select, input[type="text"], input[type="number"], input[type="email"], input[type="password"] { width: 100%; padding: 12px; margin-bottom: 15px; border-radius: 5px; border: 1px solid var(--border-color); box-sizing: border-box; font-size: 0.95em; background-color: #3a3a3c; color: var(--text-color); transition: border-color 0.2s ease, box-shadow 0.2s ease; }
        select:focus, input:not(#model-search):focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.3); outline: none; background-color: #4a4a4e; }
        button { background: linear-gradient(to right, var(--primary-color), var(--secondary-color)); color: white; padding: 12px 20px; cursor: pointer; border: none; border-radius: 5px; font-weight: bold; font-size: 1em; text-transform: uppercase; letter-spacing: 0.8px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.3s ease; margin-top: 10px; width: 100%; }
        button:hover:not(:disabled), button:focus:not(:disabled) { background: linear-gradient(to right, var(--secondary-color), var(--primary-color)); box-shadow: 0 4px 10px rgba(0,0,0,0.3); transform: translateY(-2px); }
        button:disabled { background: #555; cursor: not-allowed; opacity: 0.7; }
        #output { margin-top: 20px; padding: 15px; background-color: #1c1c1e; border: 1px solid var(--border-color); border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; max-height: 400px; overflow-y: auto; line-height: 1.6; font-family: 'Courier New', Courier, monospace; color: var(--text-color); }
        .slurm-options h3 { margin-top: 25px; font-size: 1.2em;}
        .slurm-options label { font-weight: 400; font-size: 0.85em; margin-top: 8px; }
        .endpoints-container #refresh-endpoints-button { background: linear-gradient(to right, #ffcc00, #ff9500); margin-bottom: 15px; }
        .endpoints-container #refresh-endpoints-button:hover:not(:disabled) { background: linear-gradient(to right, #ff9500, #ffcc00); }
        .endpoints-container ul { list-style-type: none; padding: 0;}
        .endpoint-item { background-color: #3a3a3cdd; border: 1px solid #4a4a4e; padding: 12px; margin-bottom: 10px; border-radius: 6px; font-size: 0.85em; line-height: 1.5; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }
        .endpoint-item strong { color: var(--secondary-color); }
        .endpoint-item a { color: var(--accent-color); text-decoration: none; font-weight: bold; word-break: break-all;}
        .endpoint-item a:hover { text-decoration: underline; color: #ffd633; }
        .footer { text-align: center; padding: 20px; background-color: #0e0e0f; color: #8e8e93; font-size: 0.9em; margin-top: auto; border-top: 3px solid var(--accent-color); }
        .icon { margin-right: 8px; font-size: 1.2em; vertical-align: middle;}
    </style>
</head>
<body>
    <div class="header"><h1><span class="icon"></span> SERAPHIM <span class="icon"></span></h1><p>Scalable Engine for Reasoning, Analysis, Prediction, Hosting, and Intelligent Modeling</p></div>
    <div class="main-container">
        <div class="form-container">
            <h3><span class="icon">⚙️</span> Deploy New vLLM Instance via Backend</h3>
            
            <label for="model-search">Search Models:</label>
            <input type="text" id="model-search" placeholder="Type to filter models..." autocomplete="off">

            <label for="model-select">Select Model:</label>
            <select id="model-select"><option value="">-- Loading models... --</option></select>
            
            <label for="service-port">Service Port (on Slurm node):</label><input type="number" id="service-port" value="8000" min="1024" max="65535"/>
            <label for="hf-token">Hugging Face Token (Optional):</label><input type="password" id="hf-token" placeholder="Needed for Llama, gated models, etc."/>
            <div class="slurm-options">
                <h3>Slurm Configuration</h3>
                <label for="job-name">Job Name:</label><input type="text" id="job-name" value="vllm_service"/>
                <label for="time-limit">Time Limit (HH:MM:SS):</label><input type="text" id="time-limit" value="23:59:59"/>
                <label for="gpus">GPUs (e.g., 1 or a100:1):</label><input type="text" id="gpus" value="1"/>
                <label for="cpus-per-task">CPUs per Task:</label><input type="number" id="cpus-per-task" value="4" min="1"/>
                <label for="mem">Memory (e.g., 32G):</label><input type="text" id="mem" value="32G"/>
                <label for="mail-user">Email Notify (Optional):</label><input type="email" id="mail-user" placeholder="your_email@example.com"/>
            </div>
            <button id="deploy-button">Deploy to Slurm via Backend</button>
            <div id="output">Configure and click deploy. Status will appear here.</div>
        </div>
        <div class="endpoints-container">
            <h3><span class="icon">📡</span> Active Deployments</h3>
            <button id="refresh-endpoints-button">Refresh Status</button>
            <div id="deployed-endpoints-list"><p><em>Loading active deployments...</em></p></div>
        </div>
    </div>
    <div class="footer">✧ SERAPHIM CORE Interface v1.5 (Searchable Models) ✧ System Online ✧</div>
    <script src="seraphim_logic.js" defer></script>
</body>
</html>
EOF_HTML
if [ $? -ne 0 ]; then echo "❌ Error: Failed to write HTML file."; exit 1; fi;
echo "✅ Frontend HTML ($HTML_FILENAME) configured."
echo ""

echo "Generating Start Script: $START_SCRIPT_TARGET_PATH"
cat > "$START_SCRIPT_TARGET_PATH" << EOF_START_SCRIPT
#!/bin/bash
SERAPHIM_DIR_START="{{SERAPHIM_DIR_PLACEHOLDER}}"
CONDA_ENV_NAME_START="{{CONDA_ENV_NAME_PLACEHOLDER}}"
CONDA_BASE_PATH_START="{{CONDA_BASE_PATH_PLACEHOLDER}}"
BACKEND_SCRIPT_START="{{BACKEND_FILENAME_PLACEHOLDER}}"
BACKEND_PORT_START={{BACKEND_PORT_PLACEHOLDER}}
FRONTEND_PORT_START={{FRONTEND_PORT_PLACEHOLDER}}

BACKEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_backend.log"
FRONTEND_LOG_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.log"
BACKEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_backend.pid"
FRONTEND_PID_FILE="\$SERAPHIM_DIR_START/seraphim_frontend.pid"

# Function to check if a port is in use
is_port_in_use() {
    local port=\$1
    if command -v ss > /dev/null; then
        ss -tulnp | grep -q ":\${port}\s"
    elif command -v netstat > /dev/null; then
        netstat -tulnp | grep -q ":\${port}\s"
    else
        echo "Warning: Neither 'ss' nor 'netstat' found. Cannot check if port \$port is in use."
        return 0 # Assume not in use if we can't check
    fi
}

echo "Starting SERAPHIM Application..."
echo "================================="

if [ -f "\$BACKEND_PID_FILE" ] && ps -p \$(cat "\$BACKEND_PID_FILE") > /dev/null; then
    echo "❌ Backend already running (PID: \$(cat "\$BACKEND_PID_FILE")). Use ./stop_seraphim.sh."
    exit 1
fi
if is_port_in_use "\$BACKEND_PORT_START"; then
    echo "❌ Error: Backend port \$BACKEND_PORT_START is already in use. Please free it or change BACKEND_PORT in install.sh and re-run."
    exit 1
fi

if [ -f "\$FRONTEND_PID_FILE" ] && ps -p \$(cat "\$FRONTEND_PID_FILE") > /dev/null; then
    echo "❌ Frontend server already running (PID: \$(cat "\$FRONTEND_PID_FILE")). Use ./stop_seraphim.sh."
    exit 1
fi
if is_port_in_use "\$FRONTEND_PORT_START"; then
    echo "❌ Error: Frontend port \$FRONTEND_PORT_START is already in use. Please free it or change FRONTEND_PORT in install.sh and re-run."
    exit 1
fi

cd "\$SERAPHIM_DIR_START" || { echo "Error: Could not navigate to \$SERAPHIM_DIR_START"; exit 1; }
echo "Activating Conda: \$CONDA_ENV_NAME_START..."
_CONDA_SH_PATH="\$CONDA_BASE_PATH_START/etc/profile.d/conda.sh"
if [ -z "\$CONDA_BASE_PATH_START" ]; then
    _FALLBACK_CONDA_BASE_PATH=\$(conda info --base 2>/dev/null)
    if [ -n "\$_FALLBACK_CONDA_BASE_PATH" ]; then _CONDA_SH_PATH="\$_FALLBACK_CONDA_BASE_PATH/etc/profile.d/conda.sh"; fi
fi
if [ ! -f "\$_CONDA_SH_PATH" ]; then echo "Error: conda.sh not found. Cannot activate."; exit 1; fi
# shellcheck source=/dev/null
. "\$_CONDA_SH_PATH"; conda activate "\$CONDA_ENV_NAME_START"
if [ "\$CONDA_DEFAULT_ENV" != "\$CONDA_ENV_NAME_START" ]; then echo "Error: Failed to activate conda env."; exit 1; fi
echo "Conda env '\$CONDA_ENV_NAME_START' activated."

echo "Starting Backend Server (port \$BACKEND_PORT_START)... Log: \$BACKEND_LOG_FILE"
nohup python "\$BACKEND_SCRIPT_START" > "\$BACKEND_LOG_FILE" 2>&1 &
_BACKEND_PID=\$!; echo \$_BACKEND_PID > "\$BACKEND_PID_FILE"
echo "Backend PID: \$_BACKEND_PID."
sleep 3; if ! ps -p \$_BACKEND_PID > /dev/null; then echo "❌ Error: Backend failed to start."; rm -f "\$BACKEND_PID_FILE"; exit 1; fi

echo "Starting Frontend Server (port \$FRONTEND_PORT_START)... Log: \$FRONTEND_LOG_FILE"
nohup python -m http.server --bind 0.0.0.0 "\$FRONTEND_PORT_START" > "\$FRONTEND_LOG_FILE" 2>&1 &
_FRONTEND_PID=\$!; echo \$_FRONTEND_PID > "\$FRONTEND_PID_FILE"
echo "Frontend PID: \$_FRONTEND_PID."
sleep 1; if ! ps -p \$_FRONTEND_PID > /dev/null; then echo "❌ Error: Frontend failed to start."; kill \$_BACKEND_PID; rm -f "\$BACKEND_PID_FILE" "\$FRONTEND_PID_FILE"; exit 1; fi

_SERVER_IP=\$(hostname -I | awk '{print \$1}' || echo "YOUR_SERVER_IP")
echo "================================="
echo "✅ SERAPHIM Application Started!"
echo "Access Frontend: http://\${_SERVER_IP}:\$FRONTEND_PORT_START"
echo "To stop: ./$STOP_SCRIPT_FILENAME"
echo "================================="
EOF_START_SCRIPT
chmod +x "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_ENV_NAME_PLACEHOLDER}}|$ESCAPED_CONDA_ENV_NAME_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{CONDA_BASE_PATH_PLACEHOLDER}}|$ESCAPED_CONDA_BASE_PATH_FOR_SED|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{BACKEND_FILENAME_PLACEHOLDER}}|$BACKEND_FILENAME|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{BACKEND_PORT_PLACEHOLDER}}|$BACKEND_PORT|g" "$START_SCRIPT_TARGET_PATH"
sed -i "s|{{FRONTEND_PORT_PLACEHOLDER}}|$FRONTEND_PORT|g" "$START_SCRIPT_TARGET_PATH"
echo "✅ Start script ($START_SCRIPT_FILENAME) created."
echo ""

echo "Generating Stop Script: $STOP_SCRIPT_TARGET_PATH"
cat > "$STOP_SCRIPT_TARGET_PATH" << EOF_STOP_SCRIPT
#!/bin/bash
SERAPHIM_DIR_STOP="{{SERAPHIM_DIR_PLACEHOLDER}}"
BACKEND_PID_FILE_STOP="\$SERAPHIM_DIR_STOP/seraphim_backend.pid"
FRONTEND_PID_FILE_STOP="\$SERAPHIM_DIR_STOP/seraphim_frontend.pid"
echo "Stopping SERAPHIM Application..."
stop_process() {
    local pid_file="\$1"; local process_name="\$2"
    if [ -f "\$pid_file" ]; then
        _PID_TO_KILL=\$(cat "\$pid_file")
        if ps -p "\$_PID_TO_KILL" > /dev/null; then
            echo "Stopping \$process_name (PID: \$_PID_TO_KILL)..."; kill "\$_PID_TO_KILL"; sleep 1
            if ps -p "\$_PID_TO_KILL" > /dev/null; then kill -9 "\$_PID_TO_KILL"; sleep 1; fi
            if ps -p "\$_PID_TO_KILL" > /dev/null; then echo "❌ Error stopping \$process_name."; else echo "✅ \$process_name stopped."; fi
        else echo "ℹ️ \$process_name (PID \$_PID_TO_KILL) not running."; fi
        rm -f "\$pid_file"
    else echo "⚠️ \$process_name PID file not found."; fi
}
stop_process "\$BACKEND_PID_FILE_STOP" "Backend Server"
stop_process "\$FRONTEND_PID_FILE_STOP" "Frontend Server"
echo "SERAPHIM Stop Attempted."
EOF_STOP_SCRIPT
chmod +x "$STOP_SCRIPT_TARGET_PATH"
sed -i "s|{{SERAPHIM_DIR_PLACEHOLDER}}|$ESCAPED_SERAPHIM_DIR_FOR_SED|g" "$STOP_SCRIPT_TARGET_PATH"
echo "✅ Stop script ($STOP_SCRIPT_FILENAME) created."
echo ""

echo "======================================================================"
echo "✅ SERAPHIM Setup Complete!"
echo "To run: cd \"$SERAPHIM_DIR\" && ./$START_SCRIPT_FILENAME"
echo "To stop: cd \"$SERAPHIM_DIR\" && ./$STOP_SCRIPT_FILENAME"
_SERVER_IP_FINAL=\$(hostname -I | awk '{print \$1}' || echo "YOUR_SERVER_IP")
echo "Access UI: http://\${_SERVER_IP_FINAL}:$FRONTEND_PORT"
echo "======================================================================"
echo "🚨 Notes: User running backend needs sbatch access. Review CORS in $BACKEND_FILENAME for production."
echo "Ensure '$MODELS_FILE_PATH' is populated with your desired models (format: model_id,Display Name)."
echo "======================================================================"
exit 0
