#!/usr/bin/env bash
# =============================================================================
# SERA 研究実行ジョブスクリプト (SLURM)
#
# 使い方:
#   sbatch scripts/run_research.sh                    # 新規実行
#   sbatch scripts/run_research.sh --resume           # 中断再開
#
# 環境変数で制御:
#   SERA_WORK_DIR    ワークスペースのパス (default: ./sera_workspace)
#   SERA_PARTITION   SLURM パーティション (default: sx40)
#   SERA_TIME        実行時間上限 (default: 20:00:00)
# =============================================================================
#SBATCH --job-name=sera-research
#SBATCH --partition=sx40
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/sera-research-%j.out
#SBATCH --error=logs/sera-research-%j.err

set -euo pipefail

# --- パス設定 ----------------------------------------------------------------
SERA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SERA_VENV_DIR:-${SERA_ROOT}/.venv}"
WORK_DIR="${SERA_WORK_DIR:-./sera_workspace}"

# --- ログディレクトリ確保 ----------------------------------------------------
mkdir -p "${SERA_ROOT}/logs"

# --- 環境 activate -----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv が見つかりません: ${VENV_DIR}"
    echo "先に GPU ノード上でセットアップを実行してください:"
    echo "  srun --partition=sx40 --time=01:00:00 bash scripts/setup_env.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# --- 環境情報記録 ------------------------------------------------------------
echo "=== SERA Research Job ==="
echo "ジョブID:    ${SLURM_JOB_ID:-local}"
echo "ホスト名:    $(hostname)"
echo "日時:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python:      $(python --version)"
echo "ワーク:      ${WORK_DIR}"
echo ""

python -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'CUDA:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)')
"
echo ""

# --- 実行 --------------------------------------------------------------------
# スクリプトに渡された引数をそのまま sera research に渡す
# 例: sbatch scripts/run_research.sh --resume
echo "=== sera research 開始 ==="
sera research --work-dir "${WORK_DIR}" "$@"
EXIT_CODE=$?

echo ""
echo "=== 完了 (exit code: ${EXIT_CODE}) ==="
exit ${EXIT_CODE}
