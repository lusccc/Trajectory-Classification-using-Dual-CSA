#!/bin/bash
path="./results/dcsa_softmax/"
mkdir -p ${path}
#cp ./results/default/RP_conv_ae_check_point.model ${path}
#cp ./results/default/ts_conv_ae_check_point.model ${path}
python Dual_Softmax_AE.py --results_path ${path}


