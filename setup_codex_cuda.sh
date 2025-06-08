#!/usr/bin/env bash
set -euo pipefail

# 1) OPENAI_API_KEY 환경변수 체크
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: Please export OPENAI_API_KEY before running."
  echo "  export OPENAI_API_KEY=sk-..."
  exit 1
fi

# 2) 빌드 디렉터리 준비
mkdir -p build
cd build

# 3) nvcc로 컴파일
#    -std=c++17, libcurl, pthread, (nlohmann/json 헤더는 상위 디렉터리 json.hpp)
nvcc -std=c++17 \
     -I . \
     codex_client.cpp \
     -o codex_client \
     -lcurl \
     -Xcompiler="-pthread"

echo "✅ Build complete. Run with ./codex_client"