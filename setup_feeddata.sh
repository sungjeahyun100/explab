#!/usr/bin/env bash
# plot_survival.sh

# 데이터 디렉토리 (필요에 따라 수정)
DATA_DIR="/home/sjh100/바탕화면/explab/dataset"
OUTPUT_DAT="counts.dat"
OUTPUT_PNG="surviving_cells.png"

# 이전 데이터 초기화
> "$OUTPUT_DAT"

# 파일을 이름 순으로 순회하며 인덱스와 11번째 줄(생존 셀 수) 추출
idx=1
for f in $(ls -v "$DATA_DIR"/*.txt); do
  # 11번째 줄을 읽어서 공백 제거
  count=$(sed -n '11p' "$f" | tr -d '[:space:]')
  echo "$idx $count" >> "$OUTPUT_DAT"
  ((idx++))
done

# gnuplot으로 그래프 생성
gnuplot <<-EOF
  set terminal pngcairo size 800,600
  set output '$OUTPUT_PNG'
  set title 'Surviving cells after stabilization'
  set xlabel 'File index'
  set ylabel 'Surviving cell count'
  plot '$OUTPUT_DAT' using 1:2 with linespoints title 'Survival Count'
EOF

echo "Generated $OUTPUT_DAT and $OUTPUT_PNG"