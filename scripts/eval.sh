#!/bin/bash

# pspnet_resnetd101b_scratch, pspnet_resnetd101b_imagenet_encoder,
# pspnet_resnetd101b_coco, pspnet_resnetd101b_coco_encoder
# pspnet_resnetd101b_voc, pspnet_resnetd101b_voc_encoder,
# resnet_unet_scratch, resnet34_unet_scratch, resnet34_unet_imagenet_encoder
model="resnet34_unet_imagenet_encoder"
gpu="0,1"

#data_fold=0 # 0 - 1 - 2 - 3 - 4
img_size=224
crop_size=224

epochs=50
defrost_epoch=7
batch_size=32

optimizer="adam"     # adam - over9000
scheduler="constant" # constant - steps - plateau - one_cycle_lr
min_lr=0.0001
max_lr=0.01
lr=0.0001 # learning_rate for conventional schedulers
dataset="mnms"

# bce_dice - bce_dice_ac - bce_dice_border - bce_dice_border_ce
criterion="bce_dice_border_ce"
weights_criterion="0.5,0.2,0.2,0.2,0.5"

normalization="reescale" # reescale - standardize

# "none" - "rotations" - "vflips" - "hflips" - "elastic_transform" - "grid_distortion"
# "shift" - "scale" - "optical_distortion" - "coarse_dropout" - "random_crops" - "downscale"
for data_augmentation in "none"
do

for data_fold in 0 1 2 3 4
do

for lr in 0.01 0.001 0.0001
do

parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/$parent_dir/datafold${data_fold}_da${data_augmentation}_scheduler_${scheduler}"

if [ $scheduler == "once_cycle_lr" ]
then
  model_path=${model_path}"_minlr${min_lr}_maxlr${max_lr}"
else
  model_path=${model_path}"_lr${lr}"
fi


echo -e "\n---- Start Evaluation ----\n"
python3 -u fold_eval.py --gpu $gpu --output_dir $model_path --epochs $epochs --defrost_epoch $defrost_epoch \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --min_lr $min_lr --max_lr $max_lr --scheduler_steps 99 --learning_rate $lr \
  --data_fold $data_fold --normalization $normalization

done


done

done



python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
