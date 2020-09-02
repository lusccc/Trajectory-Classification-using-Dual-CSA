cd ..
dataset='geolife'
for i in $(seq 0 1 4); do
  path="./results/${dataset}_lstm_fcn_softmax_exp${i}/"
  export RES_PATH=${path}
  mkdir -p ${path}
  python -m network_comparison.LSTM_FCN_Softmax_keras --dataset ${dataset} --results-path ${path}
  rm -rf ${path}/lstm_fcn_softmax.model
done