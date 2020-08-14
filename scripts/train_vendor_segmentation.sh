#!/bin/bash

# Available models:
#   -> resnet34_unet_scratch - resnet18_unet_scratch
#   -> small_segmentation_unet - small_segmentation_small_unet
#      small_segmentation_extrasmall_unet - small_segmentation_nano_unet
#   -> resnet18_pspnet_unet - resnet34_pspnet_unet

gpu="0,1"
dataset="mnms"

#data_fold=0 # 0 - 1 - 2 - 3 - 4
img_size=224
crop_size=224

epochs=80
defrost_epoch=-1
batch_size=32
label_type="mask"

lr=0.001
data_augmentation="combination_old"

optimizer="adam"     # adam - over9000
scheduler="constant" # constant - steps - plateau - one_cycle_lr

# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
criterion="bce_dice_border_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.5"

normalization="standardize" # reescale - standardize

fold_system="vendor" # vendor - patient

for model in "resnet34_unet_scratch" "resnet34_pspnet_unet"
do

data_fold="A"
data_fold_validation="B"


parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/segmentation/$parent_dir/foldBy_${fold_system}/datafold${data_fold}_foldtest${data_fold_validation}"
model_path=$model_path"_da${data_augmentation}_scheduler_${scheduler}"
model_path=${model_path}"_lr${lr}"


echo -e "\n---- Start Initial Training ----\n"
python3 -u train_segmentation.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --scheduler_steps 99 --learning_rate $lr \
  --data_fold $data_fold --normalization $normalization --fold_system $fold_system \
  --data_fold_validation $data_fold_validation --label_type $label_type


data_fold="B"
data_fold_validation="A"


parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/$parent_dir/foldBy_${fold_system}/datafold${data_fold}_foldtest${data_fold_validation}"
model_path=$model_path"_da${data_augmentation}_scheduler_${scheduler}"
model_path=${model_path}"_lr${lr}"


echo -e "\n---- Start Initial Training ----\n"
python3 -u train.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --scheduler_steps 99 --learning_rate $lr \
  --data_fold $data_fold --normalization $normalization --fold_system $fold_system \
  --data_fold_validation $data_fold_validation --label_type $label_type


done

python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
