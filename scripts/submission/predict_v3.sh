#!/bin/bash

# Usage: ./scripts/submission/predict_v3.sh "<input_data_folder_path>" "<output_results_folder>"
# Optional Usage Overlays: ./scripts/submission/predict_v3.sh "<input_data_folder_path>" "<output_results_folder>" "<overlays_out_folder"

gpu="0"
dataset="mnms_test"

model="resnet34_unet_scratch"
ckpt="/evaluationSubmission/segmentator_EntropyMnms_swa_v3_3.pt"

img_size=224
crop_size=224

normalization="standardize"
data_augmentation="none"

input_data_directory=$1
output_data_directory=$2
eval_overlays_path=${3:-"none"}

echo -e "\n---- Start Prediction ----\n"
# Flags -> --eval_overlays_path
python3 -u /predict_v1.py --gpu $gpu --dataset $dataset --img_size $img_size --crop_size $crop_size \
  --normalization $normalization --model_name $model --data_augmentation $data_augmentation --model_checkpoint $ckpt \
  --input_data_directory $input_data_directory --output_data_directory $output_data_directory \
  --eval_overlays_path $eval_overlays_path

