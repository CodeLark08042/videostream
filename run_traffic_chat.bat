@echo off
chcp 65001 >nul
echo ========================================================
echo   正在启动 Traffic Chat 后端服务和隧道
echo ========================================================

:: 1. 启动 Python 后端 (在新窗口中)
echo [1/2] 正在启动 Python 后端 (traffic_server.py)...
start "Traffic Chat Backend" cmd /k "python traffic_server.py"

:: 2. 等待几秒确保后端启动
timeout /t 3 /nobreak >nul

:: 3. 启动 Cloudflare Tunnel (当前窗口)
echo.
echo [2/2] 正在启动 Cloudflare Tunnel...
echo.
echo ==================== 重要提示 ====================
echo 隧道启动后，请在下方寻找类似这样的链接：
echo https://random-name-xxxx.trycloudflare.com
echo.
echo 请复制该链接，并替换 app.js 中的 serverUrl 地址！
echo ==================================================
echo.
.\cloudflared.exe tunnel --url http://localhost:5000
pause
