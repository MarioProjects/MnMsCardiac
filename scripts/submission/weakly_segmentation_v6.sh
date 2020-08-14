#!/bin/bash

model="resnet34_unet_scratch"
gpu="0,1"

img_size=224
crop_size=224

batch_size=32

optimizer="adam"     # adam - over9000
scheduler="steps" # constant - steps - plateau - one_cycle_lr

# bce - bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
criterion="bce_dice_border_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.5"

normalization="standardize" # reescale - standardize

label_type="mask"
fold_system="all" # vendor - patient
data_augmentation="combination_old"

criterion="bce_dice_border_haus_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.05,0.5"

dataset="mnms"
parent_dir="$model/${dataset}_from_weakly/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/submission_v6/segmentation/$parent_dir/foldBy_${fold_system}/da${data_augmentation}_scheduler_${scheduler}"
model_path=${model_path}"_lr${lr}"

echo -e "\n---- Start Training ----\n"

lr=0.002; swa_start=0; swa_freq=1; swa_lr=0.002; epochs=50
initial_checkpoint='evaluationSUbmission/segmentator_v5.pt'
python3 -u train_segmentation.py --gpu $gpu --output_dir $model_path --epochs $epochs \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation --learning_rate $lr \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --normalization $normalization --fold_system $fold_system --label_type $label_type \
  --swa_start $swa_start --swa_freq $swa_freq --swa_lr $swa_lr --scheduler_steps 999 \
  --model_checkpoint $initial_checkpoint --apply_swa




python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
