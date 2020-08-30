# dim=3 tau=8;emb=304;beta=11
cd ..

for i in $(seq 0 1 9); do
  path="./results/new_SHL_optimal_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  python ./PEDCC.py --save_dir ./data/SHL_features --dim 304
  python network_training.py --dataset SHL --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 230 --pretrained

done

