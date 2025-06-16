
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

set terminal pngcairo size 1000,600 enhanced font 'Arial,12'
set output 'loss_graph.png'
set title 'Epoch–Loss Comparison'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

set style line  1 lc rgb '#E41A1C' pt 7 ps 1 lw 2
set style line  2 lc rgb '#377EB8' pt 5 ps 1 lw 2 
set style line  3 lc rgb '#4DAF4A' pt 9 ps 1 lw 2  
set style line  4 lc rgb '#000000' pt 9 ps 1 lw 2
set style line  5 lc rgb '#FF00FF' pt 9 ps 1 lw 2
set style line  6 lc rgb '#FF7F00' pt 9 ps 1 lw 2
set style line  7 lc rgb '#984EA3' pt 9 ps 1 lw 2
set style line  8 lc rgb '#A65628' pt 9 ps 1 lw 2

plot \
  'loss_data_Xavier_Tanh--Softsign--Tanh_CrossEntropy_batch50.txt'       using 1:2 with linespoints ls 1 title 'Tanh--Softsign--Tanh', \
  'loss_data_Xavier_Softsign--Tanh--Softsign_CrossEntropy_batch50.txt'   using 1:2 with linespoints ls 2 title 'Softsign--Tanh--Softsign', \
  'loss_data_Xavier_Swish--Tanh--Softsign_CrossEntropy_batch50.txt'      using 1:2 with linespoints ls 3 title 'Swish--Tanh--Softsign', \
  'loss_data_Xavier_Tanh--Swish--Softsign_CrossEntropy_batch50.txt'      using 1:2 with linespoints ls 4 title 'Tanh--Swish--Softsign', \
  'loss_data_Xavier_LReLU--Tanh--Softsign_CrossEntropy_batch50.txt'      using 1:2 with linespoints ls 5 title 'LReLU--Tanh--Softsign', \
  'loss_data_Xavier_Softplus--Tanh--Softsign_CrossEntropy_batch50.txt'   using 1:2 with linespoints ls 6 title 'Softplus--Tanh--Softsign', \
  'loss_data_Xavier_Softplus--Tanh--Sigmoid_CrossEntropy_batch50.txt'    using 1:2 with linespoints ls 7 title 'Softplus--Tanh--Sigmoid', \
  'loss_data_Xavier_LReLU--Tanh--Sigmoid_CrossEntropy_batch50.txt'       using 1:2 with linespoints ls 8 title 'LReLU--Tanh--Sigmoid'

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
    with linespoints lw 2 pt 7 ps 1 title 'He-SELU-MSE'


unset output

exit

EOF

