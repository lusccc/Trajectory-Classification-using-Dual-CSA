cd ..
for i in $(seq 308 8 360); do
  path="./results/exp1st_SHL_t7s3e${i}/"
  mkdir -p ${path}
  each_emb_dim=$(expr ${i} / 2)
  ts_emb_dim=${each_emb_dim}
  RP_emb_dim=${each_emb_dim}
  total_emb=${i}
  # copy trained autoencoder weight files. no need to pretrain
  cp ./results/best_dual_ae_model/RP_conv_ae_check_point.model ${path}
  cp ./results/best_dual_ae_model/ts_conv_ae_check_point.model ${path}
  python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_train.npy --save_dir ./data/SHL_features --data_type train --dim ${total_emb}
  python ./PEDCC.py --features_path ./data/SHL_features/trjs_segs_features_test.npy --save_dir ./data/SHL_features --data_type test --dim ${total_emb}
  python -m network.Dual_SAE --dataset SHL --results_path ${path} --RP_emb_dim ${RP_emb_dim} --ts_emb_dim ${ts_emb_dim}
  rm -rf ${path}/RP_conv_ae_check_point.model
  rm -rf ${path}/ts_conv_ae_check_point.model
done


