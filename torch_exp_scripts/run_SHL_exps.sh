# dim=3 tau=8;emb=304;beta=11
cd ..

for i in $(seq 0 1 10); do
  path="./results/SHL_optimal_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  cp ./results/SHL_pretrained_emb304/pretrained_AE.pt ${path}
  python network_training.py --dataset SHL --results-path ${path} --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 448 --pretrained

done

