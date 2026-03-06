@echo off
chcp 65001 >nul
cd /d "%~dp0"
set DEVICE=cuda
set VIDEO_SOURCE=test1.mp4
:: To use ONNX, simply set MODEL_NAME to your .onnx file path
:: e.g. set MODEL_NAME=best.onnx
set MODEL_NAME=yolov8n.pt
set PORT=5000
python server.py
