#loss weight = 1,1,1; dim3 tau=8;
cd ..
for j in $(seq 0 1 5); do
  for i in $(seq 8 8 408); do
    path="./results/SHL_emb${i}_exp${j}/"
    export RES_PATH=${path}
    mkdir -p ${path}
    each_emb_dim=$(expr ${i} / 2)
    FS_emb_dim=${each_emb_dim}
    RP_emb_dim=${each_emb_dim}
    total_emb=${i}
    python ./PEDCC.py --save_dir ./data/SHL_features --dim ${total_emb}
    python network_training.py --dataset SHL --results-path ${path} --alpha 1 --beta 1 --gamma 1 --RP-emb-dim ${RP_emb_dim} --FS-emb-dim ${FS_emb_dim} --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 400
  done
done
