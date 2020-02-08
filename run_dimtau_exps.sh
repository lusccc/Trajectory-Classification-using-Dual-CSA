#!/bin/bash
dims=(5 6 9 9 9 9)
taus=(8 8 6 7 8 9)


for i in `seq 0 1 7`
do
  path="./results/dim${dims[${i}]}tau${taus[${i}]}/"
  mkdir -p ${path}
  cp ./results/ts_conv_ae_check_point.model ${path}

  python MF_RP_mat.py --dim ${dims[${i}]} --tau ${taus[${i}]}
  python dataset.py
  python Dual_SAE.py --results_path ${path}
done


