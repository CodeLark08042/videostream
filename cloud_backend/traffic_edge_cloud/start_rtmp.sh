#!/bin/bash

# 检查是否安装了 Nginx RTMP
if [ ! -f "/usr/local/nginx/sbin/nginx" ]; then
    echo "[Info] 首次运行，正在自动安装 Nginx RTMP 服务器..."
    if [ -f "install_nginx.sh" ]; then
        chmod +x install_nginx.sh
        ./install_nginx.sh
    else
        echo "[Error] install_nginx.sh 未找到！"
    fi
fi

# 确保 Nginx 已启动
if ! pgrep -x "nginx" > /dev/null; then
    echo "[Info] 启动 Nginx..."
    /usr/local/nginx/sbin/nginx
else
    echo "[Info] Nginx 已在运行。"
fi

# 安装 Python 依赖
if [ -f "requirements.txt" ]; then
    echo "[Info] 检查 Python 依赖..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null 2>&1
fi

# 确保使用 .pt 模型 (AutoDL 环境下 TensorRT 引擎可能不兼容)
if [ ! -f "best.pt" ]; then
    if [ -f "../best.pt" ]; then
        echo "[Info] 从上级目录复制 best.pt..."
        cp ../best.pt .
    else
        echo "[Warning] 当前目录未找到 best.pt，请确保已上传模型文件！"
    fi
fi

# 清理旧的 engine 文件防止干扰
if [ -f "best.engine" ]; then
    echo "[Info] 移除旧的 TensorRT engine 文件以强制重新导出或使用 PT..."
    rm best.engine
fi

echo "=================================================="
echo "服务准备就绪！"
echo "请在本地 OBS 中推流至以下地址:"
echo "服务器: rtmp://<你的AutoDL公网IP>:1935/live"
echo "串流密钥: stream"
echo "--------------------------------------------------"
echo "推流成功后，后端将自动开始处理..."
echo "=================================================="

# 启动 Python 后端
export VIDEO_SOURCE="rtmp://localhost/live/stream"
export PORT=6006
export DEVICE="cuda"
export MODEL_NAME="best.pt" 
python server_rtmp.py
