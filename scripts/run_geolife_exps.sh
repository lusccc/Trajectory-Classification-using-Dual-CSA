cd ..

for i in $(seq 0 1 9); do
  path="./results/exp${i}_geolife_t7s3e304a1b3y1_optimal/"
  mkdir -p ${path}
  cp ./results/best_dual_ae_model_geolife/RP_conv_ae_check_point.model ${path}
  cp ./results/best_dual_ae_model_geolife/ts_conv_ae_check_point.model ${path}
  python -m network.Dual_SAE --dataset geolife --results_path ${path} --alpha 1 --beta 3 --gamma 1 --RP_emb_dim 152 --ts_emb_dim 152
  rm -rf ${path}/RP_conv_ae_check_point.model
  rm -rf ${path}/ts_conv_ae_check_point.model
done