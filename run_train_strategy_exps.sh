#!/bin/bash
no_pre=(True True False)
no_joint=(False True False)


for i in `seq 0 1 2`
do
  path="./results/no_pre${no_pre[${i}]}no_joint${no_joint[${i}]}/"
  mkdir -p ${path}
  cp ./results/epoch_200_500/RP_conv_ae_check_point.model ${path}
  cp ./results/epoch_200_500/ts_conv_ae_check_point.model ${path}
  python Dual_SAE.py --results_path ${path} --no_pre ${no_pre[${i}]} --no_joint ${no_joint[${i}]}
done

