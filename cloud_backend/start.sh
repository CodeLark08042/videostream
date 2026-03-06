#!/bin/bash

# AutoDL 一键启动脚本
# 使用方法: bash start.sh

# 1. 确保脚本出错即停
set -e

echo "=================================================="
echo "   Traffic Chat Cloud Backend - AutoDL Launcher   "
echo "=================================================="

# 2. 检查并安装依赖 (如果还没装)
if ! python -c "import ultralytics" &> /dev/null; then
    echo "[Info] 正在安装依赖..."
    pip install -r requirements.txt
    # 安装 OpenCV 系统库 (AutoDL 必需)
    apt-get update && apt-get install -y libgl1 libglib2.0-0
else
    echo "[Info] 依赖已安装，跳过。"
fi

# 3. 设置环境变量
export DEVICE=cuda
export PORT=6006
export VIDEO_SOURCE=test1.mp4

# 4. 自动检测模型 (优先顺序: 环境变量 > best.engine > best.onnx > best.pt > server.py自动处理)
if [ -n "$MODEL_NAME" ]; then
    echo "[Info] 使用环境变量指定模型: $MODEL_NAME"
elif [ -f "best.engine" ]; then
    export MODEL_NAME=best.engine
    echo "[Info] 自动检测到 TensorRT 模型: best.engine (最快速度)"
elif [ -f "best.onnx" ]; then
    export MODEL_NAME=best.onnx
    echo "[Info] 自动检测到模型: best.onnx"
elif [ -f "best.pt" ]; then
    export MODEL_NAME=best.pt
    echo "[Info] 自动检测到模型: best.pt"
else
    echo "[Info] 未找到 best.engine/onnx/pt，将交由 Python 脚本自动搜索其他模型..."
fi

# 5. 启动服务
echo "=================================================="
echo "服务启动中... 请在 AutoDL 控制台获取 [自定义服务] 链接"
echo "访问地址示例: https://uXXXXX-xxxxx.westb.seetacloud.com:8443"
echo "=================================================="

python server.py