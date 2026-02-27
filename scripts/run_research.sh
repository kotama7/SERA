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
# SLURM実行時は SLURM_SUBMIT_DIR を使用（sbatchがスクリプトをコピーするため）
SERA_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${SERA_VENV_DIR:-${SERA_ROOT}/.venv}"
WORK_DIR="${SERA_WORK_DIR:-./sera_workspace}"

# --- ログディレクトリ確保 ----------------------------------------------------
mkdir -p "${SERA_ROOT}/logs"

# --- 環境 activate -----------------------------------------------------------
# .env ファイルから API キーを読み込み
if [ -f "${SERA_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${SERA_ROOT}/.env"
    set +a
    echo ".env loaded"
fi

if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif command -v conda &>/dev/null; then
    # conda 環境を使用（.venv がない場合のフォールバック）
    eval "$(conda shell.bash hook)"
    echo "Using conda environment"
else
    echo "ERROR: .venv も conda も見つかりません"
    exit 1
fi

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
        props = torch.cuda.get_device_properties(i)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({mem / 1024**3:.1f} GB)')
" || echo "(torch info skipped)"
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
