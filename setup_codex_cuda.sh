#!/usr/bin/env bash
set -euo pipefail

# 2) 빌드 디렉터리 준비
mkdir -p build
cd build

# 3) nvcc로 컴파일

nvcc -std=c++20 \
    -I .. \                   # 프로젝트 루트 (공통 헤더 json.hpp 등)
    -I ../src \               # src/ 안에 있는 헤더
    -I ../exp_sample\         # exp_sample/ 안에 있는 헤더 (필요시)
    ../exp_sample/GOLexp.cu\  # 컴파일할 CUDA 소스
    -o codex_exp \            # 결과 바이너리 이름
    -lcurl \                  # libcurl 링크
    -lcurand \                # CUDA 커랜드(난수) 라이브러리 링크
    -Xcompiler="-pthread"     # pthread 플래그를 호스트 컴파일러에 전달

echo "✅ Build complete. Run with ./codex_exp"