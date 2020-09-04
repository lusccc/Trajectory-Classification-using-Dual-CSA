cd ..
dataset='geolife'
for i in $(seq 0 1 9); do
  path="./results/${dataset}_dca_softmax_exp${i}/"
  mkdir -p ${path}
  export RES_PATH=${path}
done