#!/bin/bash

# Usage:
# bash train.sh <base_dir> <batch_size> <num_epochs> <lr> <model_type> <sampling_strategy>
# Example:
# bash train.sh /path/to/CodecFake/all_data_16k/ 8 20 1e-6 S_BIN AUX

# Parse arguments
base_dir="$1"
batch_size="$2"
num_epochs="$3"
lr="$4"
model_type="$5"
sampling_strategy="$6"

# Check if all arguments are provided
if [ -z "$base_dir" ] || [ -z "$batch_size" ] || [ -z "$num_epochs" ] || [ -z "$lr" ] || [ -z "$model_type" ] || [ -z "$sampling_strategy" ]; then
  echo "Usage: $0 <base_dir> <batch_size> <num_epochs> <lr> <model_type> <sampling_strategy>"
  echo "Example: $0 /path/to/CodecFake/all_data_16k/ 8 20 1e-6 S_BIN AUX"
  exit 1
fi

# Run training
CUDA_VISIBLE_DEVICES=0 python train.py \
  --base_dir="${base_dir}" \
  --metadata_path="./metadata/training/" \
  --batch_size="${batch_size}" \
  --num_epochs="${num_epochs}" \
  --lr="${lr}" \
  --save_dir="./models" \
  --model_type="${model_type}" \
  --sampling_strategy="${sampling_strategy}"


  