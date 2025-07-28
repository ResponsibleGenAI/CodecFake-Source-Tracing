#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 8 ]; then
  echo "Usage: bash train.sh <base_dir> <save_dir> <batch_size> <num_epochs> <lr> <task> <sampling_strategy> <mask_ratio>"
  echo ""
  echo "Example:"
  echo "  ./run_train.sh /path/to/data ./models_myname 12 40 1e-5 AUX AUX 0.4"
  echo ""
  echo "Arguments:"
  echo "  <base_dir>           Path to Codecfake+ CoRS dataset (called all_data_16k/)"
  echo "  <save_dir>           Directory to save model checkpoints"
  echo "  <batch_size>         Batch size (e.g., 12)"
  echo "  <num_epochs>         Number of training epochs (e.g., 40)"
  echo "  <lr>                 Learning rate (e.g., 1e-5)"
  echo "  <task>               Task type (e.g., Bin / AUX / DEC / VQ)"
  echo "  <sampling_strategy>  Sampling strategy (e.g., AUX / DEC / VQ)"
  echo "  <mask_ratio>         Mask ratio for MAE (e.g., 0.4)"
  echo ""
  exit 1
fi

# Read input arguments
base_dir="$1"
save_dir="$2"
batch_size="$3"
num_epochs="$4"
lr="$5"
task="$6"
sampling_strategy="$7"
mask_ratio="$8"

# Run training
CUDA_VISIBLE_DEVICES=0 python train.py \
  --base_dir="${base_dir}" \
  --metadata_path="./metadata/training/" \
  --batch_size="${batch_size}" \
  --num_epochs="${num_epochs}" \
  --lr="${lr}" \
  --save_dir="${save_dir}" \
  --task="${task}" \
  --sampling_strategy="${sampling_strategy}" \
  --mask_2D \
  --mask_ratio="${mask_ratio}" \
  --use_SSL_feat \
  --use_semantic \
  --use_multi_decoder \
  --load_tuned_weight \
  --tuned_weight_path="./Pretrain_weight/tuned_weight.pth"
ain_weight/tuned_weight.pth"
