#!/usr/bin/env bash
set -euo pipefail

# RKNN 服务运行脚本
# 对应 Docker 容器中的 ENTRYPOINT ["tini", "--"] CMD ["python", "-m", "immich_ml"]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 加载环境变量
if [[ -f "$PROJECT_ROOT/.env.rknn" ]]; then
    set -a
    source "$PROJECT_ROOT/.env.rknn"
    set +a
fi

# 激活虚拟环境
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
    export VIRTUAL_ENV="$VENV_DIR"
    export PATH="$VENV_DIR/bin:$PATH"
    echo "[run] 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "[run] 警告: 虚拟环境不存在，请先运行 ./prepare.sh" >&2
fi

# 设置默认环境变量
export DEVICE="${DEVICE:-rknn}"
export MACHINE_LEARNING_MODEL_ARENA="${MACHINE_LEARNING_MODEL_ARENA:-false}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# 启用 core dump
ulimit -c unlimited

# 设置 core 文件路径
echo "/tmp/core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern 2>/dev/null || true

# 如果启用调试，设置信号处理
if [[ "${RKNN_DEBUG:-}" == "1" ]]; then
    trap 'echo "Process crashed with signal $?. Core file: /tmp/core.$$"; sleep 5; gdb -p $$ -ex "thread apply all bt" -ex "detach" -ex "quit"' SIGSEGV SIGABRT
fi

# Immich 配置
export IMMICH_HOST="${IMMICH_HOST:-0.0.0.0}"
export IMMICH_PORT="${IMMICH_PORT:-3003}"
export IMMICH_WORKERS="${IMMICH_WORKERS:-1}"
export MACHINE_LEARNING_CACHE_FOLDER="${MACHINE_LEARNING_CACHE_FOLDER:-$PROJECT_ROOT/.cache}"

# 创建缓存目录
mkdir -p "$MACHINE_LEARNING_CACHE_FOLDER"

# RKNN 库路径
DOWNLOAD_DIR="$PROJECT_ROOT/.rknn_deps"
if [[ -f "$DOWNLOAD_DIR/librknnrt.so" ]]; then
    export LD_LIBRARY_PATH="$DOWNLOAD_DIR:${LD_LIBRARY_PATH:-}"
fi

echo "[run] 启动 Immich ML 服务 (RKNN 模式)..."
echo "[run] 设备: $DEVICE"
echo "[run] 主机: $IMMICH_HOST:$IMMICH_PORT"
echo "[run] 工作进程: $IMMICH_WORKERS"
echo "[run] 缓存目录: $MACHINE_LEARNING_CACHE_FOLDER"
echo ""

# 检查 uv 环境
if ! command -v uv &> /dev/null; then
    echo "[run] 错误: 未找到 uv，请先运行 ./prepare.sh" >&2
    exit 1
fi

# 使用 uv 运行 Python 模块
cd "$PROJECT_ROOT"
if [[ "${RKNN_DEBUG:-}" == "1" ]]; then
    gdb -ex "set pagination off" -ex run -ex "thread apply all bt" -ex quit --args $(which python) -m immich_ml
else
    uv run --frozen python -m immich_ml
fi