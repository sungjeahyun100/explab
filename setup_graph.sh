
cd graph/count_ver_loss/

gnuplot << 'EOF'
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'loss_with_points_Xavier.png'
set title 'Epoch–Loss graph'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

plot \
  'loss_data_Xaiver_Sigmoid_CrossEntropy_batch50.txt' using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'Xavier-Sigmoid-CrossEntropy', \
  'loss_data_Xaiver_Tanh_CrossEntropy_batch50.txt' using 1:2 \
    with linespoints lw 2 pt 5 ps 1 title 'Xavier-Tanh-CrossEntropy', \
  'loss_data_Xavier_Softsign_CrossEntropy_batch50.txt' using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'Xavier-Softsign-CrossEntropy'

unset output

set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'loss_with_points_He.png'
set title 'Epoch–Loss graph (in log scale)'
set xlabel 'Epoch'
set ylabel 'Loss'
set logscale y 10
set yrange [0.01:*]  
set grid

plot 'loss_data_He_LReLU_MSE_batch50.txt' every ::1 using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'He-LReLU-MSE',\
     'loss_data_He_ReLU_MSE_batch50.txt' every ::1 using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'He-ReLU-MSE',\
    'loss_data_He_ELU_MSE_batch50.txt' every ::1 using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'He-ELU-MSE',\
    'loss_data_He_SELU_MSE_batch50.txt' every ::1 using 1:2 \
    with linespoints lw 2 pt 7 ps 1 title 'He-SELU-MSE', \
    'loss_data_He_SELU_CrossEntropy_batch50.txt' using 1:2\
    with linespoints lw 2 pt 5 ps 1 title 'He-SELU-CrossEntropy'


unset output

exit

EOF

