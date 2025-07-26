#!/bin/bash

# Check argument count
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <base_dir> <dataset_type> <model_path> <task> <eval_output>"
  echo ""
  echo "Example:"
  echo "  ./run_inference.sh /path/to/data CoSG ./checkpoints/latest.ckpt AUX ./Result"
  echo ""
  echo "Arguments:"
  echo "  <base_dir>       Path to Codecfake+ CoRS dataset (called all_data_16k/) or CoSG dataset (called SLMdemos_16k/)"
  echo "  <dataset_type>   Dataset type: CoRS or CoSG"
  echo "  <model_path>     Path to trained model checkpoint"
  echo "  <task>           Task type: Bin / AUX / DEC / VQ"
  echo "  <eval_output>    Directory to save inference results"
  exit 1
fi

# Parse input arguments
base_dir="$1"
dataset_type="$2"
model_path="$3"
task="$4"
eval_output="$5"

# Select metadata file
if [ "$dataset_type" = "CoSG" ]; then
  txt_path="./metadata/evaluation/CoSG_ALL.txt"
elif [ "$dataset_type" = "CoRS" ]; then
  txt_path="./metadata/evaluation/CoRS_ALL.txt"
else
  echo "Error: Unsupported dataset_type '$dataset_type'. Must be 'CoSG' or 'CoRS'."
  exit 1
fi

# Run inference
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --dataset_type="${dataset_type}" \
  --metadata_path="${txt_path}" \
  --model_path="${model_path}" \
  --base_dir="${base_dir}" \
  --task="${task}" \
  --eval_output="${eval_output}" \
  --use_SSL_feat \
  --use_semantic
