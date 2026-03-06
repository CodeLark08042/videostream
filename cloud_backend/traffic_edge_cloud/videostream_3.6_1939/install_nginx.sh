#!/bin/bash
apt-get update
apt-get install -y build-essential libpcre3 libpcre3-dev libssl-dev zlib1g-dev
mkdir -p /root/nginx_src
cd /root/nginx_src
wget http://nginx.org/download/nginx-1.24.0.tar.gz
tar -zxvf nginx-1.24.0.tar.gz
git clone https://github.com/arut/nginx-rtmp-module.git
cd nginx-1.24.0
./configure --with-http_ssl_module --add-module=../nginx-rtmp-module
make && make install
cat > /usr/local/nginx/conf/nginx.conf <<CONF
worker_processes  1;
events { worker_connections 1024; }
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
    include mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;
    server {
        listen 80;
        server_name localhost;
        location / {
            root html;
            index index.html index.htm;
        }
    }
}
CONF
/usr/local/nginx/sbin/nginx
echo "Nginx Installed and Started!"
