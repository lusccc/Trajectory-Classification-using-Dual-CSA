cd ..

for i in $(seq 0 1 9); do
  path="./results/exp${i}_SHL_no_hcr_3,7_152_304_1,8,1/"
  mkdir -p ${path}
  cp ./results/best_dual_ae_model_SHL/RP_conv_ae_check_point.model ${path}
  cp ./results/best_dual_ae_model_SHL/ts_conv_ae_check_point.model ${path}
  python -m network.Dual_SAE --dataset SHL --results_path ${path} --alpha 1 --beta 8 --gamma 1 --RP_emb_dim 152 --ts_emb_dim 152
  rm -rf ${path}/RP_conv_ae_check_point.model
  rm -rf ${path}/ts_conv_ae_check_point.model
done