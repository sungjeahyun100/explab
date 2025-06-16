#!/usr/bin/env bash
find graph/count_ver_loss_deriv/single_plots -name "*.png" -delete

set -euo pipefail

cd graph/count_ver_loss_deriv/
mkdir -p single_plots

for f in deriv_*.txt; do
  model=$(basename "$f" .txt | sed 's/^deriv_//' | sed 's/_/-/g')
  gnuplot <<EOF
    # 1) 터미널 & 기본 폰트 크기 통일
    set terminal pngcairo size 800,600 font 'Arial,12'
    set output "single_plots/${model}.png"
    
    # 2) 타이틀 / 축 레이블 / 범례 폰트 크기 지정
    set title "ΔLoss per Epoch – ${model}" font 'Arial,16'
    set xlabel "Epoch"                  font 'Arial,14'
    set ylabel "ΔLoss"                  font 'Arial,14'
    set key font 'Arial,12'             # 범례 폰트
    
    # 3) 눈금(label) 크기 통일
    set tics font 'Arial,12'
    
    # 4) 격자 및 0축
    set grid lw 1 lc rgb '#cccccc'
    set xzeroaxis lw 2 lc rgb 'black' lt 1
    
    # 5) 플롯
    plot "${f}" using 1:2 \
      with linespoints lc rgb '#377EB8' pt 9 ps 1 lw 2 title "${model}"
    
    unset output
EOF
  echo "✅ single_plots/${model}.png 생성 완료"
done