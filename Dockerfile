# --- Stage 1: Build Environment ---
# 使用一个包含完整构建工具的Debian镜像作为基础
FROM debian:bookworm as builder

# 设置非交互式安装，防止apt-get卡住
ENV DEBIAN_FRONTEND=noninteractive

# 安装所有编译C++程序所需的系统依赖
# 包括：基础构建工具、cmake、OpenCV、Dlib的依赖(如libjpeg, libpng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    libopencv-dev \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /build

# 复制C++相关文件到构建环境中
COPY curve_calculate.cpp .
COPY json.hpp .
COPY dlib/ dlib/

# 编译C++应用程序
# -I/usr/include/opencv4: 指定OpenCV头文件路径
# -Idlib: 指定Dlib头文件路径
# dlib/dlib/all/source.cpp: 这是编译Dlib所必需的
# `pkg-config ...`: 自动链接OpenCV库
RUN g++ -std=c++17 -O3 -I/usr/include/opencv4 -Idlib -o process_csr_app ERF2.cpp dlib/dlib/all/source.cpp `pkg-config --cflags --libs opencv4` -lpthread -lX11


# --- Stage 2: Final Production Environment ---
# 使用一个轻量的Python镜像作为最终运行环境
FROM python:3.12-slim-bookworm

# 再次安装非编译时需要的运行时库 (OpenCV运行时库)
# 这使得最终的镜像比包含所有编译工具的镜像小得多
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 \
    libopencv-imgproc406 \
    libopencv-imgcodecs406 \
    libopencv-highgui406 \
    && rm -rf /var/lib/apt/lists/*
# 注意: 上述库的版本号(406)可能需要根据Debian:bookworm的具体情况调整

WORKDIR /app

# 从第一阶段(builder)复制已经编译好的C++程序
COPY --from=builder /build/process_csr_app .

# 复制Python应用和前端文件
COPY requirements.txt .
COPY app.py .
COPY index.html .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# Render平台会自动将外部流量导向这个端口
EXPOSE 10000

# 定义容器启动时运行的命令

CMD ["waitress-serve", "--host=0.0.0.0", "--port=10000", "app:app"]
