#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트 경로 설정
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 빌드 출력 디렉터리 생성
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

# CUDA 및 공통 소스 파일 포함하여 컴파일
nvcc -std=c++20 \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/exp_sample/GOLexp.cu" \
    "$PROJECT_ROOT/src/perceptron.cu" \
    "$PROJECT_ROOT/src/d_matrix.cu" \
    "$PROJECT_ROOT/src/database.cu" \
    -o "$BUILD_DIR/codex_exp" \
    -lcurl \
    -lcurand \
    -Xcompiler="-pthread"

echo "✅ build/codex_exp 빌드 완료"

nvcc -std=c++20 \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/exp_sample/genGOL.cu" \
    "$PROJECT_ROOT/src/d_matrix.cu" \
    "$PROJECT_ROOT/src/database.cu" \
    -o "$BUILD_DIR/genGOLdata" \
    -lcurl \
    -Xcompiler="-pthread"

echo "✅ build/genGOLdata 빌드 완료"

nvcc -std=c++20 \
    -I "$PROJECT_ROOT/src" \
    "$PROJECT_ROOT/exp_sample/pradic_GOL_count.cu" \
    "$PROJECT_ROOT/src/perceptron.cu" \
    "$PROJECT_ROOT/src/d_matrix.cu" \
    "$PROJECT_ROOT/src/database.cu" \
    -o "$BUILD_DIR/GOL_count_pradic_exp" \
    -lcurl \
    -lcurand \
    -Xcompiler="-pthread"

echo "✅ build/GOL_count_pradic_exp 빌드 완료"

