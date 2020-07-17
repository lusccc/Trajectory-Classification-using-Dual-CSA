# other parameters fixed to:
#m = 8, τ = 8, α = 1, β = 3, γ = 1
for i in `seq 16 8 192`
do
    mkdir -p "./results/emb${i}/"
    cp ./results/default/RP_conv_ae_check_point.model "./results/emb${i}/"
    cp ./results/default/ts_conv_ae_check_point.model "./results/emb${i}/"
    python Dual_SAE.py --results_path "./results/emb${i}/" --dataset geolife
done
