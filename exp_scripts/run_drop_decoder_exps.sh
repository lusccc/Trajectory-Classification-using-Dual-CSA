cd ..
dataset='geolife'
for i in $(seq 2 1 4); do
  path="./results/${dataset}_drop_decoder_exp${i}"
  mkdir ${path}
  export RES_PATH=${path}
  python network_training.py --dataset ${dataset} --results-path ${path}   --RP-emb-dim 152 --FS-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 410 --pretrain-epochs 5 --pretrained
done