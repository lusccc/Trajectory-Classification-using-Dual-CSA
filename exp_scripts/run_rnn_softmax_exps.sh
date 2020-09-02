cd ..
dataset='geolife'
for i in $(seq 0 1 4); do
  path="./results/${dataset}_rnn_softmax_exp${i}/"
  export RES_PATH=${path}
  mkdir -p ${path}
  python -m network_comparison.RNN_Softmax_keras --dataset ${dataset} --results-path ${path}
  rm -rf ${path}/rnn_softmax.model
done