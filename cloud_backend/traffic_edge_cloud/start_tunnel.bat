@echo off
:: 请将下面的端口和域名替换成你 AutoDL 的真实信息
set PORT=11036
set HOST=root@connect.westb.seetacloud.com

echo ==================================================
echo 正在建立推流隧道 (本地1936 -^> 服务器1935)...
echo 请输入 AutoDL 密码...
echo ==================================================

:: 只需要这一条隧道给 OBS 用
ssh -CNg -L 1936:localhost:1935 -p %PORT% %HOST%

:: 如果你需要第二条隧道给网页用（绕过 AutoDL 代理），可以去掉下面这行的注释
start ssh -CNg -L 6006:localhost:6006 -p %PORT% %HOST%

pause
