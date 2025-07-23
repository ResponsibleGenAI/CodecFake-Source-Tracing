#!/bin/bash

# Usage example:
# bash inference.sh CoRS /path/to/CodecFake/all_data_16k/ /path/to/model.pth S_BIN
# bash inference.sh CoSG /path/to/CodecFake/SLMdemos_16k/ /path/to/model.pth S_BIN

# Read command-line arguments
dataset_type="$1"
base_dir="$2"
model_path="$3"
model_type="$4"

# Check if all arguments are provided
if [ -z "$dataset_type" ] || [ -z "$base_dir" ] || [ -z "$model_path" ] || [ -z "$model_type" ]; then
  echo "Usage: $0 <dataset_type> <base_dir> <model_path> <model_type>"
  echo "Example: $0 CoRS /path/to/CoRS_dataset_dir /path/to/model.pth S_BIN"
  exit 1
fi

# Set txt_path based on dataset_type
if [ "$dataset_type" = "CoSG" ]; then
  txt_path="./metadata/evaluation/CoSG_ALL.txt"
elif [ "$dataset_type" = "CoRS" ]; then
  txt_path="./metadata/evaluation/CoRS_ALL.txt"
else
  echo "Unsupported dataset_type: $dataset_type"
  exit 1
fi

# Run inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --base_dir="${base_dir}" \
  --metadata_path="${txt_path}" \
  --dataset_type="${dataset_type}" \
  --model_path="${model_path}" \
  --model_type="${model_type}"




