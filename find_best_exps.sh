#!/bin/bash

for i in `seq 0 1 5`
do
  path="./results/find_best_only_joint${i}/"
  mkdir -p ${path}
  cp ./results/epoch_200_500/RP_conv_ae_check_point.model ${path}
  cp ./results/epoch_200_500/ts_conv_ae_check_point.model ${path}
  python Dual_SAE.py --results_path ${path} --no_pre True --no_joint False
done

for i in `seq 0 1 5`
do
  path="./results/find_best_pre_joint${i}/"
  mkdir -p ${path}
  cp ./results/epoch_200_500/RP_conv_ae_check_point.model ${path}
  cp ./results/epoch_200_500/ts_conv_ae_check_point.model ${path}
  python Dual_SAE.py --results_path ${path} --no_pre False --no_joint False
done




