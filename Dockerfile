# --- Stage 1: Build Environment ---
FROM debian:bookworm as builder

ENV DEBIAN_FRONTEND=noninteractive

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
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN wget http://dlib.net/files/dlib-19.24.tar.bz2
RUN tar xvf dlib-19.24.tar.bz2
RUN mv dlib-19.24 dlib

RUN mkdir -p nlohmann
RUN wget https://github.com/nlohmann/json/releases/latest/download/json.hpp -O nlohmann/json.hpp

COPY process_csr.cpp . 

# 【已修改】在 g++ 命令中增加了 -DDLIB_NO_GUI_SUPPORT 参数
RUN g++ -std=c++17 -O3 -I/usr/include/opencv4 -Idlib -DDLIB_NO_GUI_SUPPORT -o process_csr_app process_csr.cpp dlib/dlib/all/source.cpp `pkg-config --cflags --libs opencv4` -lpthread

# --- Stage 2: Final Production Environment ---
# (第二阶段完全保持不变)
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 \
    libopencv-imgproc406 \
    libopencv-imgcodecs406 \
    && rm -rf /var/lib/apt/lists/*
    # 删除了 libopencv-highgui, 因为 highgui 模块主要用于GUI

WORKDIR /app

COPY --from=builder /build/process_csr_app .
COPY requirements.txt .
COPY app.py .
COPY index.html .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=10000", "app:app"]


