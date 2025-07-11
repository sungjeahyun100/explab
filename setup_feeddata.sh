#!/usr/bin/env bash
# plot_survival.sh

# 데이터 디렉토리 (필요에 따라 수정)
DATA_DIR="/home/sjh100/바탕화면/explab/dataset"
FREQ_DATA="freq.dat"          # output data file for gnuplot
OUTPUT_PNG="surviving_cells_freq.png"  # output image
TMP_COUNTS="$(mktemp)"        # temporary file to hold raw counts

# --- Extract counts ---
shopt -s nullglob
txt_files=("$DATA_DIR"/*.txt)
if [ ${#txt_files[@]} -eq 0 ]; then
  echo "Error: No .txt files found in '$DATA_DIR'."
  exit 1
fi

> "$TMP_COUNTS"
for f in "${txt_files[@]}"; do
  # Read 11th line, strip non-digits
  count=$(sed -n '11p' "$f" | tr -dc '0-9')
  if [[ $count =~ ^[0-9]+$ ]]; then
    echo "$count"
  fi
done > "$TMP_COUNTS"

if [ ! -s "$TMP_COUNTS" ]; then
  echo "Error: No counts were extracted. Check that files have at least 11 lines."
  rm -f "$TMP_COUNTS"
  exit 1
fi

# --- Build frequency table ---
sort -n "$TMP_COUNTS" | uniq -c | awk '{print $2, $1}' > "$FREQ_DATA"
rm -f "$TMP_COUNTS"

# --- Plot using gnuplot ---
gnuplot <<-EOF
  set terminal pngcairo size 800,600
  set output '$OUTPUT_PNG'
  set title 'Frequency of Surviving Cell Counts'
  set xlabel 'Surviving cell count'
  set ylabel 'Number of files'
  set grid
  set autoscale
  plot '$FREQ_DATA' using 1:2 with linespoints title 'Frequency'
EOF

echo "Generated '$FREQ_DATA' and '$OUTPUT_PNG'."