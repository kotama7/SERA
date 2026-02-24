#!/usr/bin/env bash
# =============================================================================
# SERA 論文生成・評価ジョブスクリプト (SLURM)
#
# 使い方:
#   sbatch scripts/run_paper.sh              # Phase 7 + Phase 8
#   sbatch scripts/run_paper.sh --eval-only  # Phase 8 のみ
# =============================================================================
#SBATCH --job-name=sera-paper
#SBATCH --partition=sx40
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/sera-paper-%j.out
#SBATCH --error=logs/sera-paper-%j.err

set -euo pipefail

SERA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SERA_VENV_DIR:-${SERA_ROOT}/.venv}"
WORK_DIR="${SERA_WORK_DIR:-./sera_workspace}"

mkdir -p "${SERA_ROOT}/logs"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv が見つかりません。scripts/setup_env.sh を先に実行してください。"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "=== SERA Paper Generation ==="
echo "ジョブID: ${SLURM_JOB_ID:-local}"
echo "ホスト名: $(hostname)"
echo ""

EVAL_ONLY=false
for arg in "$@"; do
    if [ "$arg" = "--eval-only" ]; then
        EVAL_ONLY=true
    fi
done

if [ "$EVAL_ONLY" = false ]; then
    echo "--- Phase 7: 論文生成 ---"
    sera generate-paper --work-dir "${WORK_DIR}"
fi

echo "--- Phase 8: 論文評価 ---"
sera evaluate-paper --work-dir "${WORK_DIR}"

echo ""
echo "=== 完了 ==="
