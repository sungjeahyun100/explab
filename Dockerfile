# 1) Codex Universal 이미지를 베이스로 사용
FROM ghcr.io/openai/codex-universal:latest

# 2) CUDA 툴킷 설치 (Ubuntu 기반이라 apt 사용)
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
 && rm -rf /var/lib/apt/lists/*

# 3) (선택) 환경 변수로 코드 실행 환경 버전 고정
ENV CODEX_ENV_PYTHON_VERSION=3.12 \
    CODEX_ENV_NODE_VERSION=20

# 4) 워크디렉터리 설정
WORKDIR /workspace

# 컨테이너 시작 시 bash 로 진입
ENTRYPOINT ["bash"]