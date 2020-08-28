
 if [ ${i} -le 2 ]
  then
    n_features=1
  elif [ ${i} -ge 3 ] && [ ${i} -le 5 ]
  then
    n_features=2



 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ddp_test.py


python network_training.py --dataset geolife --results-path ./results/geolife_pretrained_emb304 --alpha 1 --beta 11 --gamma 1 --RP-emb-dim 152 --FS-emb-dim 152 --patience 20 --dist-url tcp://127.0.0.1:6666 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 330
python ./PEDCC.py --save_dir ./data/geolife_features --dim 304


for i in $(seq 0 1 9); do
  echo ${i}
done


