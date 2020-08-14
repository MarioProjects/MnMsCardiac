#!/bin/bash

# Training classification with Vendor A - B - C

model="resnet34_unet_scratch_classification"
gpu="0,1"
dataset="mnms"

#data_fold=0 # 0 - 1 - 2 - 3 - 4
img_size=224
crop_size=224

epochs=90
batch_size=100

optimizer="adam"
scheduler="steps"

normalization="standardize"

label_type="vendor_label_full" # vendor_label_binary - vendor_label_full
fold_system="all"
num_classes=3

# bce - ce
criterion="ce"
weights_criterion="1.0"

info_append="_1channel"
data_augmentation="combination_old"
lr=0.001

parent_dir="$model/$dataset/$optimizer/${criterion}_weights${weights_criterion}/normalization_${normalization}"
model_path="results/submission_v3/classification/$parent_dir/foldBy_${fold_system}"
model_path=$model_path"_da${data_augmentation}_scheduler_${scheduler}${info_append}"
model_path=${model_path}"_lr${lr}"

echo -e "\n---- Start Initial Training ----\n"
python3 -u train_classification.py --gpu $gpu --output_dir $model_path --epochs $epochs \
  --batch_size $batch_size --model_name $model --data_augmentation $data_augmentation \
  --img_size $img_size --crop_size $crop_size --scheduler $scheduler --optimizer $optimizer \
  --criterion $criterion --weights_criterion $weights_criterion --dataset $dataset \
  --scheduler_steps 45 80 --learning_rate $lr \
  --normalization $normalization --fold_system $fold_system --label_type $label_type \
  --num_classes $num_classes


python3 utils/slack_message.py --msg "[M&Ms Challenge 2020] Experiment finished!"
