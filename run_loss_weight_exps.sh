#!/bin/bash
#alpha=(1 0 1 1 1 1 1 1 1 1 1)
#beta=(1 1 1 1.5 2 2.5 3 3.5 4 4.5 5)
#gamma=(1 1 0 1 1 1 1 1 1 1 1)

alpha=(1)
beta=(3)
gamma=(1)


for i in `seq 0 1 0`
do
  path="./results/a${alpha[${i}]}b${beta[${i}]}y${gamma[${i}]}/"
  mkdir -p ${path}
  cp ./results/RP_conv_ae_check_point.model ${path}
  cp ./results/ts_conv_ae_check_point.model ${path}
  python Dual_SAE.py --results_path ${path} --alpha ${alpha[${i}]} --beta ${beta[${i}]} --gamma ${gamma[${i}]}
done

