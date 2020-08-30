# dim=3 tau=8;emb=304;beta=11
cd ..
for i in $(seq 0 1 4); do
  path="./results/new_SHL_only_joint_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  python ./PEDCC.py --save_dir ./data/SHL_features --dim 304
  python network_training.py --dataset SHL --results-path ${path}   --RP-emb-dim 152 --FS-emb-dim 152  --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --gpu 0 -b 130 --training-strategy only_joint_training
done

