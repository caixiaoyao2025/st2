# ============================================================
# 一键装 Ollama + 从 ModelScope 下 qwen2.5:14b
# 在 conda activate /caixiaoyao/envs/biomni311 后执行
# ============================================================

# ---- Step 1: 装 Ollama 二进制 ----
echo ">>> Installing Ollama binary..."
curl -fsSL https://ollama.com/install.sh | sh

# ---- Step 2: 设置持久路径 ----
export OLLAMA_MODELS=/caixiaoyao/ollama_models
mkdir -p $OLLAMA_MODELS

# ---- Step 3: 启动 Ollama 服务 ----
ollama serve &
sleep 3

# ---- Step 4: 从 ModelScope 下载 ----
echo ">>> Installing modelscope..."
pip install modelscope -q

python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'qwen/Qwen2.5-14B-Instruct-GGUF',
    cache_dir='/caixiaoyao/ollama_models/modelscope'
)
print('ModelScope download complete')
"

# ---- Step 5: 导入 Ollama ----
GGUF_FILE=$(find /caixiaoyao/ollama_models/modelscope -name "*q4_k_m*" -type f | head -1)
echo "GGUF file: $GGUF_FILE"

ollama create qwen2.5:14b -f /dev/stdin << EOF
FROM $GGUF_FILE
EOF

# ---- Step 6: 验证 ----
ollama list
echo "All done!"
