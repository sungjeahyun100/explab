
cd graph/count_ver_loss/

gnuplot << 'EOF'
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'loss_with_points.png'
set title 'Epoch–Loss with Data Points'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

plot \
  'loss_data_Xaiver_Sigmoid_CrossEntropy_batch50.txt' using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'Xavier-Sigmoid', \
  'loss_data_Xaiver_Tanh_CrossEntropy_batch50.txt' using 1:2 \
    with linespoints lw 2 pt 5 ps 1 title 'Xavier-Tanh', \
  'loss_data_He_LReLU_MSE_batch50.txt' every ::2 using 1:2 \
    with linespoints lw 2 pt 9 ps 1 title 'He-LReLU'


exit

EOF

