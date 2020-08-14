#!/bin/bash

# Usage: ./scripts/predict.sh "<input_data_folder_path>" "<output_results_folder>"

# pspnet_resnetd101b_scratch, pspnet_resnetd101b_imagenet_encoder,
# pspnet_resnetd101b_coco, pspnet_resnetd101b_coco_encoder
# pspnet_resnetd101b_voc, pspnet_resnetd101b_voc_encoder,
# resnet_unet_scratch, resnet34_unet_scratch, resnet34_unet_imagenet_encoder
model="resnet34_unet_imagenet_encoder"
gpu="0,1"
dataset="mnms_test"

ckpt="checkpoints/mnms.pt"

input_data_directory=$1
output_data_directory=$2
eval_overlays_path=${3:-"none"}

img_size=224
crop_size=224

normalization="reescale" # reescale - standardize

# "none" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion"
# "shift" - "scale" - "optical_distortion" - "coarse_dropout" - "random_crops" - "downscale"
data_augmentation="none"

echo -e "\n---- Start Prediction ----\n"
# Flags -> --eval_overlays_path
python3 -u predict.py --gpu $gpu --dataset $dataset --img_size $img_size --crop_size $crop_size \
  --normalization $normalization --model_name $model --data_augmentation $data_augmentation --model_checkpoint $ckpt \
  --input_data_directory $input_data_directory --output_data_directory $output_data_directory \
  --eval_overlays_path $eval_overlays_path

