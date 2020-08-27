cd ..
for i in $(seq 0 1 9); do
  path="./results/SHL_lstm_fcn_exp${i}/"
  export RES_PATH=${path}
  mkdir -p ${path}
  python -m network_comparison.LSTM_FCN_Softmax_keras --dataset SHL --results-path ${path}
  rm -rf ${path}/lstm_fcn_softmax.model
done