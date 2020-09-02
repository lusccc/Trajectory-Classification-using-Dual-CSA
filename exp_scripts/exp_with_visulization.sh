cd ..
i=0
dataset='geolife'
path="./results/${dataset}_emb_visualization_exp${i}"
export RES_PATH=${path}
python network_training.py --dataset ${dataset} --results-path ${path}   --RP-emb-dim 152 --FS-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 310  --visualize-emb 1 --training-strategy normal_only_pretraining --pretrain-epochs 100


dataset='SHL'
path="./results/${dataset}_emb_visualization_exp${i}"
export RES_PATH=${path}
python network_training.py --dataset ${dataset} --results-path ${path}   --RP-emb-dim 152 --FS-emb-dim 152  --patience 15 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 410  --visualize-emb 1 --training-strategy normal_only_pretraining --pretrain-epochs 100
