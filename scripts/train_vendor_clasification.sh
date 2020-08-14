#!/bin/bash

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet
model="resnet34_unet_scratch_classification"
gpu="0,1"
dataset="mnms"

#data_fold=0 # 0 - 1 - 2 - 3 - 4
img_size=224
crop_size=224

epochs=90
defrost_epoch=7
batch_size=100

optimizer="adam"     # adam - over9000
scheduler="steps" # constant - steps - plateau - one_cycle_lr

normalization="standardize" # reescale - standardize

fold_system="patient" # vendor - patient
label_type="vendor_label_full" # vendor_label_binary - vendor_label_full
num_classes=3

# bce - ce
criterion="ce"
weights_criterion="1.0"

info_append="_1channel"
data_augmentation="combination_old"
lr=0.002

for data_fold in 0 1 2 3 4
do

parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/classification/$parent_dir/foldBy_${fold_system}/trfold${data_fold}"
model_path=$model_path"_da${data_augmentation}_scheduler_${scheduler}${info_append}"
model_path=${model_path}"_lr${lr}"


echo -e "\n---- Start Initial Training ----\n"
python3 -u train_classification.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --scheduler_steps 30 60 --learning_rate $lr \
  --data_fold $data_fold --normalization $normalization --fold_system $fold_system --label_type $label_type \
  --num_classes $num_classes


done

python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
