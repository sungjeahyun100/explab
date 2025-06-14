FROM ghcr.io/openai/codex-universal:latest

# 시스템 패키지 설치: CUDA 툴킷 + build-essential + gnuplot
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      build-essential \
      gnuplot \
 && rm -rf /var/lib/apt/lists/*

# Codex 환경 변수
ENV CODEX_ENV_PYTHON_VERSION=3.12 \
    CODEX_ENV_NODE_VERSION=20

# 기본 C/C++ 컴파일러 설정
ENV CC=gcc \
    CXX=g++

WORKDIR /workspace
ENTRYPOINT ["bash"]