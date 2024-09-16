#!/bin/bash
#SBATCH -c 16        # Number of cores (-c)
#SBATCH -t 1-10:10     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner,kempner_requeue # Partition to submit to 
#SBATCH --requeue
#SBATCH --mem-per-cpu=10000
#SBATCH --gpus-per-node=1
#SBATCH -J totem_fish_$1 #job_name
#SBATCH -o ./output/myoutput_gbt_%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./output/myerrors_gbt_%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --account=kempner_krajan_lab

module load python/3.10.9-fasrc01
export LD_LIBRARY_PATH=/n/home05/sjohnsonyu/.conda/envs/totem-3:$LD_LIBRARY_PATH
source activate totem-3

gpu=0
seq_len=64
fish_mode=$1  # all, a, b, c, d
loss_fn=$2    # bce, weighted_bce
mask_type=$3  # 'random', 'end', 'end_single'
iters=$4      # number of iterations
compression_factor=4
embedding_dim=64
num_codewords=256
train_seed=12
pred_len=0
mask_ratio=0.25
batch_size=8192
data_name=custom
root_path_name=/n/home05/sjohnsonyu/TOTEM
data_path_name="imputation_and_forecasting_data/real_fish_day_12ms_eods_only_12hrs_fish_${fish_mode}.csv"
data_storage_dir="/n/holylabs/LABS/krajan_lab/Users/sjohnsonyu/elephantfish-talking-storage/totem_vars/imputation/data/"
dataset_name="real_fish_day_12ms_eods_only_12hrs_seq_len_${seq_len}_fish_${fish_mode}"
model_upper_dir_name="lossfn_${loss_fn}_masktype_${mask_type}_maskratio_${mask_ratio}/"
model_sub_dir_name="CD${embedding_dim}_CW${num_codewords}_CF${compression_factor}_BS${batch_size}_ITR${iters}_seed${train_seed}_LF${loss_fn}_MT${mask_type}_MR${mask_ratio}"

export PYTHONPATH=$root_path_name

# Determine class_weights based on loss_fn
if [ "$loss_fn" == "bce" ]; then
    class_weights="[1,1]"
elif [ "$loss_fn" == "weighted_bce" ]; then
    class_weights="[1,10]"
else
    echo "Invalid loss_fn: $loss_fn"
    exit 1
fi

# Create the JSON configuration file
json_dir="imputation/scripts"
mkdir -p $json_dir
json_filename="${json_dir}/${dataset_name}_${loss_fn}_${mask_type}.json"

cat > $json_filename << EOL
{
    "vqvae_config": {
        "model_name": "vqvae",
        "model_save_name": "vqvae",
        "pretrained": false,
        "learning_rate": 1e-3,
        "num_training_updates": ${iters},
        "block_hidden_size": 128,
        "num_residual_layers": 2,
        "res_hidden_size": 64,
        "embedding_dim": ${embedding_dim},
        "num_embeddings": ${num_codewords},
        "commitment_cost": 0.25,
        "compression_factor": ${compression_factor},
        "dataset": "${dataset_name}",
        "loss_fn": "${loss_fn}",
        "mask_type": "${mask_type}",
        "class_weights": ${class_weights}
    },
    "comet_config": {
        "api_key": "l78zaphU6Io9DXDQyKhwbzte0",
        "project_name": "${dataset_name}",
        "workspace": "sjohnsonyu"
    }
}
EOL

# Check if preprocessed data already exists
if [ ! -d "${data_storage_dir}${dataset_name}" ]; then
    echo "Preprocessed data not found. Running preprocessing..."
    # Run the preprocessing script
    python -u imputation/preprocess_data.py \
      --random_seed $train_seed \
      --data $data_name \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --label_len 0 \
      --enc_in 10 \
      --gpu $gpu \
      --save_path "${data_storage_dir}${dataset_name}"
else
    echo "Preprocessed data already exists at ${data_storage_dir}${dataset_name}, skipping preprocessing."
fi

# Run the training script with the generated JSON config
python imputation/train_vqvae.py \
      --config_path ${json_filename} \
      --model_init_num_gpus $gpu \
      --data_init_cpu_or_gpu cpu \
      --comet_log \
      --comet_tag pipeline \
      --comet_name "vqvae_real_fish_day_12ms_fish_${fish_mode}" \
      --save_path "imputation/saved_models/${dataset_name}/${model_upper_dir_name}/" \
      --base_path $data_storage_dir \
      --batchsize $batch_size \
      --mask_ratio $mask_ratio \
      --revined_data 'False' \
      --seed $train_seed \
      --mask_type $mask_type

for mask_ratio_test in 0.25
do
python imputation/imputation_performance.py \
      --dataset $dataset_name \
      --trained_vqvae_model_path "imputation/saved_models/${dataset_name}/${model_upper_dir_name}/${model_sub_dir_name}/checkpoints/final_model.pth" \
      --compression_factor $compression_factor \
      --gpu $gpu \
      --base_path $data_storage_dir \
      --mask_ratio $mask_ratio_test
done

echo "done"
