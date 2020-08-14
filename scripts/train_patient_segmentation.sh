#!/bin/bash

# Available models -> resnet34_unet_scratch
model="resnet34_unet_scratch"
gpu="0,1"
dataset="mnms"

#data_fold=0 # 0 - 1 - 2 - 3 - 4
img_size=224
crop_size=224

epochs=70
batch_size=32

optimizer="adam"     # adam - over9000
scheduler="steps" # constant - steps - plateau - one_cycle_lr

# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce - bce_dice_border_haus_ce
criterion="bce_dice_border_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.5"

normalization="standardize" # reescale - standardize

label_type="mask"
fold_system="patient" # vendor - patient
data_augmentation="combination_old"

for data_fold in "0" "1" "2" "3" "4"
do

for lr in 0.001
do

parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/segmentation/$parent_dir/foldBy_${fold_system}/datafold${data_fold}_da${data_augmentation}_scheduler_${scheduler}"
model_path=${model_path}"_lr${lr}"


echo -e "\n---- Start Initial Training ----\n"
python3 -u train_segmentation.py --gpu $gpu --output_dir $model_path --epochs $epochs \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer --learning_rate $lr \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset --scheduler_steps 45 60 \
  --data_fold $data_fold --normalization $normalization --fold_system $fold_system --label_type $label_type

done

done


python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
