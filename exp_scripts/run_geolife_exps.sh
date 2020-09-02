# dim=3 tau=8;emb=304;
cd ..

for i in $(seq 0 1 10); do
  path="./results/new_geolife_optimal_exp${i}"
  export RES_PATH=${path}
  mkdir -p ${path}
#  cp ./results/geolife_pretrained_emb304/pretrained_AE.pt ${path}
  python ./PEDCC.py --save_dir ./data/geolife_features --dim 304
  python network_training.py --dataset geolife --results-path ${path}  --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --gpu 1 --world-size 1 --rank 0 -b 200

done

