#!/usr/bin/env bash
set -euo pipefail

# 2) 빌드 디렉터리 준비
mkdir -p build

# 3) nvcc로 컴파일

nvcc -std=c++20 \
  -I ./src \
  exp_sample/GOLexp.cu \
  src/perceptron.cu \
  src/d_matrix.cu \
  src/database.cu \
  src/chess.cu \
  -o build/codex_exp \
  -lcurl \
  -lcurand \
  -Xcompiler="-pthread"

echo "✅ Build complete. Run with ./codex_exp"
