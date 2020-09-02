python network_training.py --dataset SHL --results_path ./results/DDP_test --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --dist-url tcp://10.16.9.152:6666 --dist-backend nccl --multiprocessing-distributed --world-size 2 --rank 0

python network_training.py --dataset SHL --results_path ./results/DDP_test --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --dist-url tcp://10.16.9.152:6666 --dist-backend nccl --multiprocessing-distributed --world-size 2 --rank 1

export RES_PATH=./results/DDP_test



export RES_PATH=./results/SHL_111_pretrained_model
python network_training.py --dataset SHL --results_path ./results/SHL_111_pretrained_model --alpha 1 --beta 1 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --patience 30 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 320
python network_training.py --dataset SHL --results_path ./results/SHL_111_pretrained_model --alpha 1 --beta 1 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --patience 30 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 384

for i in $(seq 0 1 5); do
  echo ${i}
done

python network_training.py --dataset SHL --results_path ./results/SHL_111_pretrained_model --alpha 1 --beta 1 --gamma 1 --RP_emb_dim 152 --FS_emb_dim 152 --patience 30  --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 384

python network_training.py --dataset SHL --results_path ./results/test --alpha 1 --beta 1 --gamma 1 --RP_emb_dim 4 --FS_emb_dim 4 --patience 2 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 400 --epochs 5
