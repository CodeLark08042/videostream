# 云服务器部署指南 (Cloud Deployment Guide)

## 1. 准备云服务器
推荐配置：
- **操作系统**: Ubuntu 22.04 LTS (推荐) 或 Windows Server
- **GPU**: NVIDIA T4 / A10 (强烈推荐，否则 YOLO 会很慢)
- **内存**: 8GB 以上
- **端口**: 开放 TCP 5000 (用于服务)

## 2. 上传代码
将 `cloud_backend` 文件夹内的所有文件上传到服务器。
确保包含：
- `server.py`
- `requirements.txt`
- `yolov8n.pt` (模型文件)
- `test1.mp4` (如果是测试)

## 3. 安装环境 (Ubuntu 示例)
```bash
# 更新系统
sudo apt update && sudo apt install -y python3-pip ffmpeg libsm6 libxext6

# 安装依赖
pip3 install -r requirements.txt
```

## 4. 运行服务
### 方式 A：直接运行 (测试用)
```bash
python3 server.py
```

### 方式 B：后台运行 (生产用)
使用 `nohup` 或 `systemd`。
```bash
# 设置环境变量 (可选)
export DEVICE=cuda        # 使用 GPU
export PORT=5000          # 端口
export VIDEO_SOURCE=test1.mp4  # 视频源，也可以填 RTSP 流地址

# [新] 使用 ONNX 模型 (支持自定义模型)
# 1. 将你的 .onnx 文件 (例如 best.onnx) 上传到同级目录
# 2. 修改 MODEL_NAME 环境变量
export MODEL_NAME=best.onnx

# 启动服务
nohup python3 server.py > server.log 2>&1 &
```

## 5. 对接小程序
1. 获取云服务器的 **公网 IP** (例如 `1.2.3.4`)。
2. 确保防火墙已开放 5000 端口。
3. 在小程序 `app.js` 中修改 `serverUrl`：
   ```javascript
   serverUrl: "http://1.2.3.4:5000"
   ```
   *(注意：如果小程序正式发布，必须配置 HTTPS 域名，开发版可以直接用 IP)*

## 6. 高级配置 (RTSP 摄像头)
如果你有真实的交通摄像头，只需修改环境变量：
```bash
export VIDEO_SOURCE="rtsp://admin:password@192.168.1.100:554/stream"
```
这样云服务器就会直接拉取摄像头的画面进行分析。
