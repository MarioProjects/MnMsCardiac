#!/bin/bash

# Usage: ./scripts/submission/predict_v7.sh "<input_data_folder_path>" "<output_results_folder>"
# Optional Usage Overlays: ./scripts/submission/predict_v7.sh "<input_data_folder_path>" "<output_results_folder>" "<overlays_out_folder"

gpu="0"
dataset="mnms_test"

segmentator_model="resnet34_unet_scratch"
discriminator_model="resnet34_unet_scratch_classification"

seg_ckpt="/evaluationSubmission/segmentator_swa_v2.pt"
disc_ckpt="/evaluationSubmission/discriminator_v3.pt"

img_size=224
crop_size=224

normalization="standardize"
data_augmentation="none"

input_data_directory=$1
output_data_directory=$2
eval_overlays_path=${3:-"none"}

entropy_lambda=0.9
blur_lambda=0.0001
unblur_lambda=0.0001
gamma_lambda=0.0001
target="B"
out_threshold=0.1
max_iters=200

echo -e "\n---- Start Prediction ----\n"
# Flags -> --eval_overlays_path
python3 -u /predict_v2.py --gpu $gpu --dataset $dataset --img_size $img_size --crop_size $crop_size \
  --normalization $normalization --segmentator_model_name $segmentator_model --data_augmentation $data_augmentation \
  --input_data_directory $input_data_directory --output_data_directory $output_data_directory \
  --eval_overlays_path $eval_overlays_path --segmentator_checkpoint $seg_ckpt --discriminator_checkpoint $disc_ckpt \
  --discriminator_model_name $discriminator_model --target $target --entropy_lambda $entropy_lambda \
  --max_iters $max_iters --out_threshold $out_threshold --add_unblur_param --unblur_lambda $unblur_lambda \
  --add_blur_param --blur_lambda $blur_lambda --add_gamma_param --gamma_lambda $gamma_lambda

