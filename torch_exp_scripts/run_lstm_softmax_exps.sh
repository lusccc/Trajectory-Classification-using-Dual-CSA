cd ..
for i in $(seq 0 1 9); do
  path="./results/SHL_lstm_softmax_exp${i}/"
  export RES_PATH=${path}
  mkdir -p ${path}
  python -m network_comparison.LSTM_Softmax_keras --dataset SHL --results-path ${path}
  rm -rf ${path}/lstm_softmax.model
done