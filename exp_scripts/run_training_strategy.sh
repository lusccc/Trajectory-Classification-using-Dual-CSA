# dim=3 tau=8;emb=304;
dataset='geolife'
cd ..
for i in $(seq 0 1 4); do
#  path="./results/${dataset}_only_joint_exp${i}"
  path="./results/${dataset}_no_pre_joint_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  python ./PEDCC.py --save_dir ./data/${dataset}_features --dim 304
  python network_training.py --dataset ${dataset} --results-path ${path}   --RP-emb-dim 152 --FS-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 310 --training-strategy no_pre_joint_training
done

