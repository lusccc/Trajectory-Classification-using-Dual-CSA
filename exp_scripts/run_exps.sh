# dim=3 tau=8;emb=304;
cd ..
dataset='geolife'

for i in $(seq 1 1 4); do
  path="./results/${dataset}_manual_select_pretrain_epochs5_dynamic_loss_weight_1,1,1_to_0,1,0_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
  cp ./pretrained_AE.pt ${path}
  python network_training.py --dataset ${dataset} --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 230  --pretrained

done

