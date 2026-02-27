#!/bin/bash
#SBATCH --partition=sx40
#SBATCH --time=10:00:00
#SBATCH --job-name=sera-research
#SBATCH --output=/home/t-kotama/workplace/SERA/logs/sera_pipeline_%j.out
#SBATCH --error=/home/t-kotama/workplace/SERA/logs/sera_pipeline_%j.err

set -e

cd /home/t-kotama/workplace/SERA
source .venv/bin/activate
source .env
export HF_HOME=/home/t-kotama/.cache/huggingface

echo "=== SERA Full Pipeline (Phase 2-8) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Devices:', torch.cuda.device_count())"

# Phase 2-6: Research loop + Phase 7-8: Paper (Phase 0 already done)
echo ""
echo "=== Phase 2-8: Research + Paper ==="
sera research --work-dir ./sera_workspace --skip-phase0

echo ""
echo "=== Pipeline Complete ==="
echo "End: $(date)"
