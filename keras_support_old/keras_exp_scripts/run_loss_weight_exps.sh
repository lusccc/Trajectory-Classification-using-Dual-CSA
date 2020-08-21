#!/bin/bash
#alpha=(1 0 1 1 1 1 1 1 1 1 1 1 0 1 0)
#beta=(1 1 1 1.5 2 2.5 3 3.5 4 4.5 5 1 1 1 1)
#gamma=(1 1 0 1 1 1 1 1 1 1 1 1 0 0 1)

#alpha=(1 0 1 0)
#beta=(1 1 1 1)
#gamma=(1 0 0 1)

alpha=(1 1 1 1 1 1)
beta=(1 2 3 4 5 6)
gamma=(1 1 1 1 1 1)

cd ..
for j in $(seq 0 1 4);do
  for i in $(seq 0 1 5); do
    path="./results/exp${j}_SHL_t7s3e304a${alpha[${i}]}b${beta[${i}]}y${gamma[${i}]}/"
    mkdir -p ${path}
    cp ./results/best_dual_ae_model_SHL/RP_conv_ae_check_point.model ${path}
    cp ./results/best_dual_ae_model_SHL/ts_conv_ae_check_point.model ${path}
    python -m network.Dual_SAE --dataset SHL --results_path ${path} --alpha ${alpha[${i}]} --beta ${beta[${i}]} --gamma ${gamma[${i}]} --RP_emb_dim 152 --ts_emb_dim 152
    rm -rf ${path}/RP_conv_ae_check_point.model
    rm -rf ${path}/ts_conv_ae_check_point.model
  done
done
