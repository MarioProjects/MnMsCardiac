#!/bin/bash

#out_threshold=0.5
#max_iters=100
target='A'  # A - B - C - equal

#entropy_lambda=0.1
#l1_lambda=5.0  # --add_l1
#blur_lambda=0.00001  # --add_blur_param
#unblur_lambda=0.00001  # --add_unblur_param
#gamma_lambda=0.00001  # --add_gamma_param

#l1_lambda=5.0  # --add_l1
#blur_lambda=0.000001  # --add_blur_param
#unblur_lambda=0.000001  # --add_unblur_param
#gamma_lambda=0.00001  # --add_gamma_param

# --verbose --generate_images

for max_iters in 150
do

for out_threshold in 0.5 0.25 0.1
do

for entropy_lambda in 0.1 0.5 0.9
do

for lambdas in 0.01 0.0001 0.00001
do

blur_lambda=$lambdas
unblur_lambda=$lambdas
gamma_lambda=$lambdas

python3 -u entropy_backward.py --generate_images --target $target --entropy_lambda $entropy_lambda \
                             --max_iters $max_iters --out_threshold $out_threshold \
                             --add_unblur_param --unblur_lambda $unblur_lambda \
                             --add_blur_param --blur_lambda $blur_lambda \
                             --add_gamma_param --gamma_lambda $gamma_lambda
                             #--add_l1 --l1_lambda $l1_lambda

done

done

done

done