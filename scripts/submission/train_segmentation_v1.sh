#!/bin/bash

# Training with Vendor A and B

model="resnet34_unet_scratch"
gpu="0,1"
dataset="mnms"

img_size=224
crop_size=224

epochs=100
batch_size=32

lr=0.001
data_augmentation="combination_old"

optimizer="adam"     # adam - over9000
scheduler="steps" # constant - steps - plateau - one_cycle_lr

criterion="bce_dice_border_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.5"

normalization="standardize"
label_type="mask"
fold_system="all"

parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/submission_v1/segmentation/$parent_dir/foldBy_${fold_system}/"
model_path=$model_path"da${data_augmentation}_scheduler_${scheduler}_lr${lr}"


echo -e "\n---- Start Initial Training ----\n"
python3 -u train_segmentation.py --gpu $gpu --output_dir $model_path --epochs $epochs \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation --learning_rate $lr \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset --scheduler_steps 55 89 \
  --normalization $normalization --fold_system $fold_system --label_type $label_type


lr=0.00256; swa_start=0; swa_freq=1; swa_lr=0.00256; epochs=45
initial_checkpoint=$model_path'/model_'$model'_last.pt'
python3 -u train_segmentation.py --gpu $gpu --output_dir $model_path --epochs $epochs \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation --learning_rate $lr \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --normalization $normalization --fold_system $fold_system --label_type $label_type \
  --swa_start $swa_start --swa_freq $swa_freq --swa_lr $swa_lr --scheduler_steps 999 \
  --model_checkpoint $initial_checkpoint --apply_swa

python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
