#!/bin/bash
#alpha=(1 0 1 1 1 1 1 1 1 1 1 1 0 1 0)
#beta=(1 1 1 1.5 2 2.5 3 3.5 4 4.5 5 1 1 1 1)
#gamma=(1 1 0 1 1 1 1 1 1 1 1 1 0 0 1)

#alpha=(1 0 1 0)
#beta=(1 1 1 1)
#gamma=(1 0 0 1)

cd ..

for i in `seq 0 1 14`
do
  path="./results/exp1st_SHL_a${alpha[${i}]}b${beta[${i}]}y${gamma[${i}]}/"
  mkdir -p ${path}
  cp ./results/epoch_200_500/RP_conv_ae_check_point.model ${path}
  cp ./results/epoch_200_500/ts_conv_ae_check_point.model ${path}
  python Dual_SAE.py --results_path ${path} --alpha ${alpha[${i}]} --beta ${beta[${i}]} --gamma ${gamma[${i}]}
done

