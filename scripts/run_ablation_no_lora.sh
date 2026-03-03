#!/bin/bash
#SBATCH --partition=sx40
#SBATCH --time=10:00:00
#SBATCH --job-name=sera-no_lora
#SBATCH --output=/home/t-kotama/workplace/SERA/logs/sera_no_lora_%j.out
#SBATCH --error=/home/t-kotama/workplace/SERA/logs/sera_no_lora_%j.err

set -e
cd /home/t-kotama/workplace/SERA
source .venv/bin/activate
set -a; source .env; set +a
export HF_HOME=/home/t-kotama/.cache/huggingface

echo "=== SERA Ablation: no_lora ==="
echo "Start: $(date)"
nvidia-smi | grep -E "GPU|CUDA|Memory"

sera research --work-dir ./sera_workspace_no_lora --skip-phase0
echo "Done: $(date)"
