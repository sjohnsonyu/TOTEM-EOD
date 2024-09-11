gpu=0
# seq_len=96
# seq_len=720
seq_len=640
root_path_name=/n/home05/sjohnsonyu/TOTEM
data_path_name=imputation_and_forecasting_data/real_fish_day_12ms_eods_only.csv
data_name=custom
# random_seed=2021
# random_seed=2024
random_seed=640
pred_len=0

export PYTHONPATH=$root_path_name

# python -u imputation/save_notrevin_notrevinmasked_revinx_revinxmasked.py\
#   --random_seed $random_seed \
#   --data $data_name \
#   --root_path $root_path_name \
#   --data_path $data_path_name \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --label_len 0 \
#   --enc_in 10 \
#   --gpu $gpu \
#   --save_path "/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/imputation/data/real_fish_day_12ms_eods_only_seq_len_640"


# gpu=1
# for seed in 11
# do
# for mask_ratio in 0.5 # we only train 1 model at 0.5 masking for all imputation percentages
# do
# python imputation/train_vqvae.py \
#   --config_path imputation/scripts/real_fish_day_12ms_eods_only.json \
#   --model_init_num_gpus $gpu \
#   --data_init_cpu_or_gpu cpu \
#   --comet_log \
#   --comet_tag pipeline \
#   --comet_name vqvae_real_fish_day_12ms \
#   --save_path "imputation/saved_models/real_fish_day_12ms_eods_only_seq_len_640/mask_ratio_"$mask_ratio"/"\
#   --base_path "/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/imputation/data"\
#   --batchsize 8192 \
#   --mask_ratio $mask_ratio \
#   --revined_data 'False' \
#   --seed $seed
# done
# done

compression_factor=4
num_codewords=256

for seed in 11
do
for mask_ratio_test in 0.25
# for mask_ratio_test in 0.125 0.25 0.375 0.5
do
python imputation/imputation_performance.py \
  --dataset real_fish_day_12ms_eods_only_seq_len_640 \
  --trained_vqvae_model_path "imputation/saved_models/real_fish_day_12ms_eods_only_seq_len_640/mask_ratio_0.5/CD64_CW${num_codewords}_CF${compression_factor}_BS8192_ITR15000_seed11_maskratio0.5/checkpoints/final_model.pth" \
  --compression_factor $compression_factor \
  --gpu 0 \
  --base_path "/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/imputation/data" \
  --mask_ratio $mask_ratio_test
done
done
echo "done"