# --- Stage 1: Build Environment ---
FROM debian:bookworm as builder

ENV DEBIAN_FRONTEND=noninteractive

# 【已修改】在原来的基础上增加了 wget 和 bzip2 用于下载和解压
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN wget http://dlib.net/files/dlib-19.24.tar.bz2

RUN tar xvf dlib-19.24.tar.bz2

RUN mv dlib-19.24 dlib


RUN wget https://github.com/nlohmann/json/releases/latest/download/json.hpp -O json.hpp


COPY curve_calculate.cpp .



RUN g++ -std=c++17 -O3 -I/usr/include/opencv4 -Idlib -o process_csr_app ERF2.cpp dlib/dlib/all/source.cpp `pkg-config --cflags --libs opencv4` -lpthread -lX11


FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 \
    libopencv-imgproc406 \
    libopencv-imgcodecs406 \
    libopencv-highgui406 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/process_csr_app .
COPY requirements.txt .
COPY app.py .
COPY index.html .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=10000", "app:app"]
