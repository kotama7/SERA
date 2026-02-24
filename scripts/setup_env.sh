#!/usr/bin/env bash
# =============================================================================
# SERA 環境セットアップスクリプト
#
# GPU ノード上で .venv を作成し、CUDA 対応の PyTorch + SERA をインストールする。
#
# 使い方:
#   # ログインノードから srun 経由で GPU ノード上で実行（推奨）
#   srun --partition=sx40 --time=01:00:00 bash scripts/setup_env.sh
#
#   # または sbatch でジョブとして投入
#   sbatch --partition=sx40 --time=01:00:00 scripts/setup_env.sh
#
#   # 既に GPU ノード上にいる場合は直接実行
#   bash scripts/setup_env.sh
#
# 重要: ログインノード（GPU なし）で pip install すると PyTorch が CUDA を
# 検出できず、torch.cuda.is_available() = False になる場合がある。
# 必ず GPU のある計算ノード上でこのスクリプトを実行すること。
# =============================================================================
set -euo pipefail

# --- 設定 -------------------------------------------------------------------
SERA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SERA_VENV_DIR:-${SERA_ROOT}/.venv}"
PYTHON="${SERA_PYTHON:-python3}"
CUDA_VERSION="${SERA_CUDA_VERSION:-}"  # 空 = 自動検出

# --- GPU 確認 ----------------------------------------------------------------
echo "=== SERA 環境セットアップ ==="
echo "ホスト名: $(hostname)"
echo "日時: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "--- GPU 情報 ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo ""
    echo "WARNING: nvidia-smi が見つかりません。GPU が利用できない可能性があります。"
    echo "         GPU ノード上で実行しているか確認してください。"
    echo "         例: srun --partition=sx40 --time=01:00:00 bash scripts/setup_env.sh"
    echo ""
fi

# --- CUDA バージョン自動検出 --------------------------------------------------
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        # ドライババージョンから対応 CUDA ツールキットを推定
        DETECTED_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [ -n "$DETECTED_CUDA" ]; then
            echo "検出された CUDA バージョン: ${DETECTED_CUDA}"
            # CUDA バージョンから PyTorch の index URL を決定
            CUDA_MAJOR=$(echo "$DETECTED_CUDA" | cut -d. -f1)
            CUDA_MINOR=$(echo "$DETECTED_CUDA" | cut -d. -f2)
            CUDA_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
        else
            echo "CUDA バージョンを検出できませんでした。デフォルト (cu124) を使用します。"
            CUDA_TAG="cu124"
        fi
    else
        echo "GPU が検出されません。CPU 版 PyTorch をインストールします。"
        CUDA_TAG="cpu"
    fi
else
    CUDA_TAG="cu${CUDA_VERSION//./}"
fi

echo "PyTorch CUDA ターゲット: ${CUDA_TAG}"

# --- .venv 作成 --------------------------------------------------------------
echo ""
echo "--- .venv 作成: ${VENV_DIR} ---"
if [ -d "$VENV_DIR" ]; then
    echo ".venv が既に存在します。再利用します。"
    echo "クリーン再作成する場合: rm -rf ${VENV_DIR} && bash scripts/setup_env.sh"
else
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo ".venv を作成しました。"
fi

# --- activate ----------------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Python: $(python --version) ($(which python))"

# --- pip 更新 ----------------------------------------------------------------
echo ""
echo "--- pip 更新 ---"
pip install --upgrade pip setuptools wheel

# --- PyTorch インストール (CUDA 対応) ----------------------------------------
echo ""
echo "--- PyTorch インストール (${CUDA_TAG}) ---"
if [ "$CUDA_TAG" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# --- SERA + 依存パッケージのインストール -------------------------------------
echo ""
echo "--- SERA インストール ---"
pip install -e "${SERA_ROOT}[dev,slurm]"

# --- 検証 --------------------------------------------------------------------
echo ""
echo "=== 検証 ==="
python -c "
import torch
print(f'torch version:        {torch.__version__}')
print(f'CUDA available:       {torch.cuda.is_available()}')
print(f'CUDA version (torch): {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU count:            {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA is NOT available!')
    print('  GPU ノード上でこのスクリプトを実行したか確認してください。')
"

echo ""
python -c "
import sera
print(f'sera import:          OK')
"

echo ""
python -c "
try:
    import transformers, peft, trl, accelerate, safetensors
    print(f'transformers:         {transformers.__version__}')
    print(f'peft:                 {peft.__version__}')
    print(f'trl:                  {trl.__version__}')
    print(f'accelerate:           {accelerate.__version__}')
    print(f'safetensors:          {safetensors.__version__}')
except ImportError as e:
    print(f'ERROR: {e}')
"

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "使い方:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  sera --help"
echo ""
echo "GPU ノードでの研究実行:"
echo "  sbatch scripts/run_research.sh"
echo "  # または"
echo "  srun --partition=sx40 --time=20:00:00 bash -c 'source ${VENV_DIR}/bin/activate && sera research'"
