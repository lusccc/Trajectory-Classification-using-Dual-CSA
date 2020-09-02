# dim=3 tau=8;emb=304
alpha=(1 1 1 1 1 1 1 1 1 1 1 1 1 1)
beta=(1 2 3 4 5 6 7 8 9 10 11 12 13 14)
gamma=(1 1 1 1 1 1 1 1 1 1 1 1 1 1)
cd ..
for j in $(seq 0 1 4); do
  for i in $(seq 10 1 13); do
    path="./results/SHL_loss${alpha[${i}]},${beta[${i}]},${gamma[${i}]}_exp${j}/"
    export RES_PATH=${path}
    mkdir -p ${path}
    cp ./results/SHL_pretrained_emb304/pretrained_AE.pt ${path}
    python network_training.py --dataset SHL --results-path ${path} --alpha ${alpha[${i}]} --beta ${beta[${i}]} --gamma ${gamma[${i}]} --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 448 --pretrained
    rm -rf ${path}/pretrained_AE.pt
    rm -rf ${path}/Dual_CSA.pt
  done
done
