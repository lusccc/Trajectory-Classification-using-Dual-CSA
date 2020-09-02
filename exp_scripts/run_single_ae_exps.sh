# dim=3 tau=8;emb=152;
dataset='geolife'
cd ..
for i in $(seq 0 1 4); do
  path="./results/${dataset}_only_RP_AE_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  python ./PEDCC.py --save_dir ./data/${dataset}_features --dim 152
  python network_training.py --dataset ${dataset} --network CSA_RP --results-path ${path}   --RP-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 300
done
for i in $(seq 0 1 4); do
  path="./results/${dataset}_only_FS_AE_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  python ./PEDCC.py --save_dir ./data/${dataset}_features --dim 152
  python network_training.py --dataset ${dataset} --network CSA_FS --results-path ${path}   --FS-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 2000
done