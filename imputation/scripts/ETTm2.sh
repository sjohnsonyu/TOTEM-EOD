gpu=1
seq_len=96
root_path_name=/n/home05/sjohnsonyu/TOTEM
data_path_name=ETTm2.csv
data_name=ETTm2
random_seed=2021
pred_len=0

python -u imputation/save_notrevin_notrevinmasked_revinx_revinxmasked.py\
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7\
  --gpu $gpu\
  --save_path "imputation/data/ETTm2"

gpu=1

for seed in 2021 13 1
do
for mask_ratio in 0.5 # we only train 1 model at 0.5 masking for all imputation percentages
do
python imputation/train_vqvae.py \
  --config_path imputation/scripts/ETTm2.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_ETTm2 \
  --save_path "imputation/saved_models/ETTm2/mask_ratio_"$mask_ratio"/"\
  --base_path "imputation/data"\
  --batchsize 8192 \
  --mask_ratio $mask_ratio \
  --revined_data 'False' \
  --seed $seed
done
done

for seed in 2021
do
for mask_ratio_test in 0.125 0.25 0.375 0.5
do
python imputation/imputation_performance.py \
  --dataset ETTm2 \
  --trained_vqvae_model_path "imputation/saved_models/ETTm2/ <fill in right path> /checkpoints/final_model.pth" \
  --compression_factor 4 \
  --gpu 0 \
  --base_path "imputation/data" \
  --mask_ratio $mask_ratio_test
done
done