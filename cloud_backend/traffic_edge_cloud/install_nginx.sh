#!/bin/bash
echo "[Info] 开始安装 Nginx RTMP..."
apt-get update
apt-get install -y build-essential libpcre3 libpcre3-dev libssl-dev zlib1g-dev git

echo "[Info] 下载 Nginx 源码和 RTMP 模块..."
cd /tmp
if [ ! -d "nginx-1.24.0" ]; then
    wget http://nginx.org/download/nginx-1.24.0.tar.gz
    tar -zxvf nginx-1.24.0.tar.gz
fi

if [ ! -d "nginx-rtmp-module" ]; then
    git clone https://github.com/arut/nginx-rtmp-module.git
fi

echo "[Info] 编译安装 Nginx..."
cd nginx-1.24.0
./configure --with-http_ssl_module --add-module=../nginx-rtmp-module
make -j$(nproc)
make install

echo "[Info] 配置 Nginx RTMP..."
cat > /usr/local/nginx/conf/nginx.conf <<EOF
worker_processes  1;
events {
    worker_connections  1024;
}
rtmp {
    server {
        listen 1935;
        chunk_size 4096;
        application live {
            live on;
            record off;
        }
    }
}
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;
    server {
        listen       80;
        server_name  localhost;
        location / {
            root   html;
            index  index.html index.htm;
        }
    }
}
EOF

echo "[Info] 启动 Nginx..."
/usr/local/nginx/sbin/nginx

echo "[Info] Nginx RTMP 安装完成！"
