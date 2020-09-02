cd ..
export RES_PATH=./results/test1
python ./PEDCC.py --save_dir ./data/SHL_features --dim 8
python network_training.py --dataset SHL --results-path ./results/test1 --alpha 1 --beta 1 --gamma 1 --RP-emb-dim 4 --FS-emb-dim 4 --patience 2 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 300 --epochs 2
export RES_PATH=./results/test2
python ./PEDCC.py --save_dir ./data/SHL_features --dim 16
python network_training.py --dataset SHL --results-path ./results/test2 --alpha 1 --beta 1 --gamma 1 --RP-emb-dim 8 --FS-emb-dim 8 --patience 2 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 400


#python network_training.py --dataset SHL --results-path ./results/test1 --alpha 1 --beta 1 --gamma 1 --RP-emb-dim 4 --FS-emb-dim 4 --patience 2 --dist-url tcp://10.16.9.152:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 300 --epochs 2 --world-size 2
#python network_training.py --dataset SHL --results-path ./results/test1 --alpha 1 --beta 1 --gamma 1 --RP-emb-dim 4 --FS-emb-dim 4 --patience 2 --dist-url tcp://10.16.9.152:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 1 -b 300 --epochs 2 --world-size 2


export RES_PATH=./results/SHL_pretrained_emb304
python ./PEDCC.py --save_dir ./data/SHL_features --dim 304
python network_training.py --dataset SHL --results-path ./results/SHL_pretrained_emb304 --RP-emb-dim 152 --FS-emb-dim 152 --patience 30 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 460 --only-pre


export RES_PATH=./results/aaa
python network_training.py --dataset SHL --results-path ./results/aaa --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 440