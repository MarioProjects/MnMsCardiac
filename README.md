# M&Ms Challenge 2020

The CMR images have been segmented by experienced clinicians from the respective institutions, including contours 
for the left (LV) and right ventricle (RV) blood pools, as well as for the left ventricular myocardium (MYO). 
Labels are: 1 (LV), 2 (MYO) and 3 (RV)

## Motivation

In the recent years, many machine/deep learning models have been proposed to accurately segment cardiac structures 
in magnetic resonance imaging. However, when these models are tested on unseen datasets acquired from distinct 
MRI scanners or clinical centres, the segmentation accuracy can be greatly reduced.

The M&Ms challenge aims to contribute to the effort of building generalisable models that can be applied consistently 
across clinical centres. Furthermore, M&Ms will provide a reference dataset for the community to build and assess 
future generalisable models in CMR segmentation.

## Environment Setup

To use the code, the user needs to set te environment variable to access the data. At your ~/.bashrc add:
```shell script
export MMsCardiac_DATA_PATH='/path/to/data/M&MsData/'
```

Also, the user needs to to pre-install a few packages:
```shell script
$ pip install wheel setuptools
$ pip install -r requirements.txt
$ pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torchcontrib~=0.0.2
```

### Data preparation

#### Train csv

You can generate train csv for dataloaders using `python3 preprocess/generate_train_df.py`.
```shell script
usage: generate_train_df.py [-h] [--meta_graphs]

M&Ms 2020 Challenge - Training info generation

optional arguments:
  -h, --help     show this help message and exit
  --meta_graphs  Generate train meta information graphs
```

#### Data Refactor

Load each volume to extract only 1 slice is time consuming. To solve this, save each slice in numpy arrays:
`python3 preprocess/dataloader_refactor.py`


#### Global Training Mean and STD
You can easily get global mean and std from labeled training samples using `python3 preprocess/get_mean_std.py`.

## Data Description

The challenge cohort is composed of 350 patients with hypertrophic and dilated cardiomyopathies 
as well as healthy subjects. All subjects were scanned in clinical centres in three different 
countries (Spain, Germany and Canada) using four different magnetic resonance 
scanner vendors (Siemens, General Electric, Philips and Canon).

|                Hospital                | Num. studies | Country |
|:--------------------------------------:|:------------:|:-------:|
|         Clinica Sagrada Familia        |      50      |  Spain  |
|  Hospital de la Santa Creu i Sant Pau  |      50      |  Spain  |
|      Hospital Universitari Dexeus      |      50      |  Spain  |
|         Hospital Vall d'Hebron         |      100     |  Spain  |
|     McGill University Health Centre    |      50      |  Canada |
| Universitätsklinikum Hamburg-Eppendorf |      50      | Germany |

### Training set (150+25 studies)

The training set will contain 150 annotated images from two different MRI vendors (75 each) and 25 unannotated 
images from a third vendor. The CMR images have been segmented by experienced clinicians from the respective 
institutions, including contours for the left (LV) and right ventricle (RV) blood pools, as well as for the 
left ventricular myocardium (MYO). Labels are: 1 (LV), 2 (MYO) and 3 (RV).

### Testing set (200 studies)

The 200 test cases correspond to 50 new studies from each of the vendors provided in the training set and 
50 additional studies from a fourth unseen vendor, that will be tested for model generalizability. 
20% of these datasets will be used for validation and the rest will be reserved for testing and ranking participants.

### Standard Operating Procedure (SOP) for data annotation

In order to build a useful dataset for the community we have decided to build on top of
[ACDC MICCAI 2017](https://ieeexplore.ieee.org/document/8360453) challenge SOP and correct our contours accordingly.

In particular, clinical contours have been corrected by two in-house annotators that had to agree on the final result. 
These annotators followed these rules:

  - LV and RV cavities must be completely covered, with papillary muscles included.
  - No interpolation of the LV myocardium must be performed at the base.
  - RV must have a larger surface in end-diastole compared to end-systole and avoid the pulmonary artery.
  
The main difficulty and source of disagreement is the exact RV form in basal slices.

## Results

Using ACDC checkpoint:

Average -> 0.7397 -> 0.9933 (background), 0.6931 (LV), 0.5624 (MYO), 0.71(RV)

Calculated using resnet34_unet_imagenet_encoder, Adam and constant learning rate. Fold metrics are calculated
using mean of averaged iou and dice values. Only mnms data.

|                        Method                           | Normalization | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:-------------------------------------------------------:|---------------|:------:|:------:|--------|--------|--------|--------|
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.01     |    Reescale   | 0.7958 | 0.8272 | 0.8064 | 0.8107 | 0.8220 | 0.8124 |
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.001    |    Reescale   | 0.8163 | 0.8384 | 0.8382 | 0.8336 | 0.8498 | 0.8352 |
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.0001   |    Reescale   | 0.8066 | 0.8359 | 0.8235 | 0.8281 | 0.8310 | 0.8250 |
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.01     |   Standardize | 0.7711 | 0.7745 | 0.7993 | 0.8248 | 0.7791 | 0.7897 |
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.001    |   Standardize | 0.8058 | 0.8324 | 0.8322 | 0.8138 | 0.8433 | 0.8254 |
| bce_dice_border_ce -> 0.4,0.4,0.1,0.3,0.6 - lr 0.0001   |   Standardize | 0.7970 | 0.8382 | 0.8212 | 0.8313 | 0.8344 | 0.8244 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.01     |    Reescale   | 0.7977 | 0.8150 | 0.8053 | 0.8188 | 0.8212 | 0.8116 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.001    |    Reescale   | 0.8184 | 0.8400 | 0.8339 | 0.8408 | 0.8469 | 0.8360 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.0001   |    Reescale   | 0.8096 | 0.8377 | 0.8230 | 0.8286 | 0.8316 | 0.8261 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.01     |   Standardize | 0.7842 | 0.8373 | 0.8254 | 0.8333 | 0.8318 | 0.8224 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.001    |   Standardize | 0.8235 | 0.8556 | 0.7736 | 0.8477 | 0.8598 | 0.8320 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5 - lr 0.0001   |   Standardize | 0.8221 | 0.8494 | 0.8349 | 0.8453 | 0.8503 | 0.8404 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.01   |    Reescale   | 0.7783 | 0.8101 | 0.8041 | 0.8021 | 0.8331 | 0.8055 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.001  |    Reescale   | 0.8162 | 0.8378 | 0.8330 | 0.8322 | 0.8456 | 0.8329 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.0001 |    Reescale   | 0.7971 | 0.8328 | 0.8065 | 0.8251 | 0.8291 | 0.8181 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.01   |   Standardize | 0.7893 | 0.7775 | 0.7257 | 0.8152 | 0.8162 | 0.7847 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.001  |   Standardize | 0.8091 | 0.8367 | 0.8204 | 0.8215 | 0.8436 | 0.8262 |
| bce_dice_border_ce -> 0.3,0.4,0.2,0.05,0.65 - lr 0.0001 |   Standardize | 0.7320 | 0.8234 | 0.7945 | 0.8245 | 0.8173 | 0.7983 |
| bce_dice_ce -> 0.5,0.3,0.2,0.65 - lr 0.001              |   Standardize | 0.7962 | 0.8384 | 0.8157 | 0.8053 | 0.8181 | 0.8147 |
| bce_dice_ce -> 0.5,0.3,0.2,0.65 - lr 0.0001             |   Standardize | 0.7915 | 0.8398 | 0.8148 | 0.8291 | 0.8244 | 0.8199 |

Principal conclusions: bce_dice_border_ce with 0.5,0.2,0.2,0.2,0.5 - lr 0.001/0.0001 - standardize.

Now, using lr 0.001, standardize and bce_dice_border_ce with 0.5,0.2,0.2,0.2,0.5, explore data augmentation.
Without data augmentation score 0.8360.

|   Data Augmentation   | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:---------------------:|:------:|:------:|--------|--------|--------|--------|
|   Vertical flip       | 0.8004 | 0.8273 | 0.8176 | 0.8074 | 0.8386 | 0.8182 |
|   Horizontal flip     | 0.8032 | 0.8225 | 0.8226 | 0.8244 | 0.8318 | 0.8209 |
|   Random Crops        | 0.8137 | 0.8376 | 0.8208 | 0.8283 | 0.7876 | 0.8181 |
|   Shift               | 0.8117 | 0.8240 | 0.8222 | 0.8330 | 0.8307 | 0.8243 |
|   Downscale           | 0.7949 | 0.8192 | 0.8166 | 0.8219 | 0.8384 | 0.8181 |
|   Elastic Transform   | 0.7991 | 0.8425 | 0.8274 | 0.8213 | 0.8408 | 0.8262 |
|   Rotations           | 0.8158 | 0.8426 | 0.8255 | 0.8290 | 0.8524 | 0.8330 |
|   Grid Distortion     | 0.8028 | 0.8361 | 0.7864 | 0.8275 | 0.8231 | 0.8151 |
|   Optical Distortion  | 0.7705 | 0.8418 | 0.8255 | 0.7996 | 0.8354 | 0.8145 |

### Competition Models

#### *Bala 1*

Using standardization, data augmentation combination old and bce_dice_border_ce with 0.5,0.2,0.2,0.2,0.5.
Resnet34 Unet with lr 0.001 and adam optimizer.

|             Method                | Fold 0 | Fold 1 | Fold 2 | Fold 3 |  Mean  |
|:---------------------------------:|:------:|:------:|--------|--------|--------|
| weakly -> labeled                 | 0.8286 | 0.8596 | 0.8505 | 0.8540 | 0.8482 |
| combined -> labeled               | 0.8271 | 0.8473 | 0.8424 | 0.8573 | 0.8435 |


#### *Bala 2*

Using standardization, data augmentation combination old and bce_dice_border_ce with 0.5,0.2,0.2,0.2,0.5

|             Method                | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:---------------------------------:|:------:|:------:|--------|--------|--------|--------|
| Resnet34 Unet lr 0.001            | 0.8092 | 0.8257 | 0.8115 | 0.8293 | 0.8276 | 0.8207 |


### Not Pretrained Model

Folding by patient.

|                        Method                           | Normalization | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:-------------------------------------------------------:|---------------|:------:|:------:|--------|--------|--------|--------|
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.65 - lr 0.01    |   Standardize | 0.7873 | 0.8263 | 0.8004 | 0.8195 | 0.7616 | 0.7990 |
| bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.65 - lr 0.001   |   Standardize | 0.7741 | 0.7879 | 0.7743 | 0.7883 | 0.8071 | 0.7863 |


## Update: 11/06/2020 Meeting

Changes and ideas: 

  - [x] Use 2 folds grouping by vendor (A vs. B), instead of _n_ grouping by patient. Then error analysis by vendor
  - [x] Since is not permited the use of pre-trained models, try smaller architectures
  - [ ] Create convolutional network that learns to distinguish if an image comes from vendor A or vendor B. ¿Works?
    - If works then we can create a DCGAN trying to apply a initial transformation to fool the discriminator and
     do something like normalize the input images! **Note**: Do not add vendor C in CNN classification step since
     we will use it for validate our GAN later.
  - [ ] Self-Supervised Learning for unseen vendor C

## Folding by Vendor Resuts 
#### (Wrong folding, no train subpartition/patients to compare)

Normalization by reescale. Criterion bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5.

|                 Method                  |     DA      | A -> B | B -> A |  Mean  |
|:---------------------------------------:|:-----------:|:------:|:------:|--------|
|    resnet18_pspnet_unet - lr 0.001      |    None     | 0.7573 | 0.7121 | 0.7346 |
|    resnet18_pspnet_unet - lr 0.0001     |    None     | 0.6838 | 0.5532 | 0.6185 |
|    resnet18_pspnet_unet - lr 0.001      | Combination | 0.7612 | 0.6793 | 0.7202 |
|    resnet18_pspnet_unet - lr 0.0001     | Combination | 0.6982 | 0.5580 | 0.6281 |
|    resnet18_unet_scratch - lr 0.001     |    None     | 0.7498 | 0.6835 | 0.7166 |
|    resnet18_unet_scratch - lr 0.0001    |    None     | 0.6779 | 0.4997 | 0.5888 |
|    resnet18_unet_scratch - lr 0.001     | Combination | 0.7421 | 0.6627 | 0.7023 |
|    resnet18_unet_scratch - lr 0.0001    | Combination | 0.7588 | 0.6281 | 0.6934 |
|    resnet34_unet_scratch - lr 0.001     |    None     | 0.7649 | 0.6313 | 0.6980 |
|    resnet34_unet_scratch - lr 0.0001    |    None     | 0.7189 | 0.6273 | 0.6731 |
|    resnet34_unet_scratch - lr 0.001     | Combination | 0.7673 | 0.6530 | 0.7101 |
|    resnet34_unet_scratch - lr 0.0001    | Combination | 0.7707 | 0.6128 | 0.6917 |
|    nano_unet - lr 0.001                 |    None     | 0.5035 | 0.4284 | 0.4659 |
|    nano_unet - lr 0.0001                |    None     | 0.4432 | 0.2821 | 0.3626 |
|    nano_unet - lr 0.001                 | Combination | 0.4871 | 0.4771 | 0.4821 |
|    nano_unet - lr 0.0001                | Combination | 0.4310 | 0.2187 | 0.3248 |

General conclusions: 

  - Models can extract more information and thus make better predictions when training with Vendor 'A'
    and then testing on 'B'. GAN should approximate images to Vendor A?
  - lr 0.001 works better than lower ones.
  - Not clear difference using data augmentation and without apply it...
  - Intermediate models size, resnet18_pspnet_unet, performs better than bigger ones and smaller ones.


#### 11 random patients to compare

Criterion bce_dice_border_ce -> 0.5,0.2,0.2,0.2,0.5. Using resnet18_pspnet_unet.

|  Normalization  | Data Augmentation | Learning Rate | A -> B | B -> A |  Mean  |
|:---------------:|:-----------------:|:-------------:|:------:|:------:|--------|
|   Reescale      | Combination (Old) |     0.001     | 0.7328 | 0.6915 | 0.7121 |
|   Standardize   | Combination (Old) |     0.001     | 0.7601 | 0.6704 | 0.7152 |
|   Reescale      | Combination (Old) |     0.005     | 0.6593 | 0.4914 | 0.5753 |
|   Standardize   | Combination (Old) |     0.005     | 0.7499 | 0.6342 | 0.6920 |
|   Reescale      | Combination       |     0.001     | 0.7502 | 0.7014 | 0.7258 |
|   Standardize   | Combination       |     0.001     | 0.7561 | 0.6723 | 0.7142 |
|   Reescale      | Combination       |     0.005     | 0.7370 | 0.5143 | 0.6257 |
|   Standardize   | Combination       |     0.005     | 0.7123 | 0.6826 | 0.6975 |
|   Reescale      | None              |     0.001     | 0.7462 | 0.7283 | 0.7372 |
|   Standardize   | None              |     0.001     | 0.7668 | 0.6312 | 0.6990 |
|   Reescale      | None              |     0.005     | 0.7098 | 0.6280 | 0.6689 |
|   Standardize   | None              |     0.005     | 0.7606 | 0.6604 | 0.7105 |


General conclusions: 

  - When using Vendor A as training set, generalizes better to Vendor B cases.

## Classification: Vendor 'A' - 'B' Discriminator

Using resnet18_pspnet_classification model. Adam with bce. 60 epochs and *0.1 steps as 25 and 50.
Img size 224x224. fold_system="patient" & label_type="vendor_label". Normalization standardize. Learning rate 0.001.

| Data Augmentation | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:-----------------:|:------:|:------:|--------|--------|--------|--------|
| None              | 0.9954 | 0.9726 | 1.0000 | 0.9878 | 0.9970 | 0.9906 |
| Combination       | 0.9954 | 0.9771 | 0.9985 | 1.0000 | 0.9939 | 0.9930 |

## Classification: Vendor 'A' - 'B' - 'C' Discriminator

Adam with bce. 80 epochs and *0.1 steps as 25 and 60.
Img size 224x224. fold_system="patient" & label_type="vendor_label".  Normalization standardize. Learning rate 0.001.
Data Augmentation combination (old).

| Model             | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |  Mean  |
|:-----------------:|:------:|:------:|--------|--------|--------|--------|
| resnet34_pspnet   | 0.9954 | 0.9726 | 1.0000 | 0.9878 | 0.9970 | 0.9906 |
| resnet34_pspnet   | 0.9954 | 0.9771 | 0.9985 | 1.0000 | 0.9939 | 0.9930 |
| resnet34_unet     | 0.9910 | 0.9871 | 1.0000 | 0.9740 | 0.9805 | 0.9865 |

## Discriminator Entropy backwards 'A' - 'B' - 'C'

Using gradient gamma 0.99, max iterations 250, standardize normalization. Segmentator Training with 'A'.
Baseline: 0.7799 IOU on B.

|   Out threshold   |    Target   |    More     |    B   |
|:-----------------:|:-----------:|:-----------:|:------:|
|       0.01        |      A      |    ----     | 0.7827 |
|       0.01        |      A      |    L1 2.0   | 0.7825 |
|       0.01        |      A      |    L1 5.0   | 0.7827 |
|       0.01        |      A      |    L1 10.0  | 0.7829 |
|       0.01        |    Equal    |    ----     | 0.7713 |
|       0.01        |    Equal    |    L1 2.0   | 0.7723 |
|       0.01        |    Equal    |    L1 5.0   | 0.7725 |
|       0.01        |    Equal    |    L1 10.0  | 0.7744 |
|       0.001       |      A      |    ----     | 0.7827 |
|       0.001       |      A      |    L1 2.0   | 0.7826 |
|       0.001       |      A      |    L1 5.0   | 0.7827 |
|       0.001       |      A      |    L1 10.0  | 0.7828 |
|       0.001       |    Equal    |    ----     | 0.7713 |
|       0.001       |    Equal    |    L1 2.0   | 0.7723 |
|       0.001       |    Equal    |    L1 5.0   | 0.7725 |
|       0.001       |    Equal    |    L1 10.0  | 0.7744 |
|       0.0001      |      A      |    ----     | 0.7827 |
|       0.0001      |      A      |    L1 2.0   | 0.7826 |
|       0.0001      |      A      |    L1 5.0   | 0.7828 |
|       0.0001      |      A      |    L1 10.0  | 0.7828 |
|       0.0001      |    Equal    |    ----     | 0.7713 |
|       0.0001      |    Equal    |    L1 2.0   | 0.7723 |
|       0.0001      |    Equal    |    L1 5.0   | 0.7725 |
|       0.0001      |    Equal    |    L1 10.0  | 0.7744 |

* Problem with low out thresholds... Waste all iterations and stops.

## Discriminator Entropy backwards 'A' - 'B' - 'C' / With blur, unblur and gamma

|   Out threshold   |    Entropy  |     Blur    |    Unblur   |    Gamma    |    Target   |    Iters    |    B   |
|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------:|
|       0.5         |      0.0    |   0.01      |   0.01      |   0.01      |      A      |    100      | 0.7770 |
|       0.5         |      0.0    |   0.0001    |   0.0001    |   0.0001    |      A      |    100      | 0.7786 |
|       0.5         |      0.0    |   0.000001  |   0.000001  |   0.000001  |      A      |    100      | 0.7779 |

# 7 July

### Hausdorff loss tests

Mean average values for 5 folds. Data combination old. Lr 0.001 with resnet_unet_scratch.

|   Hausdorff Weight   | IOU A  | IOU B  | DICE A | DICE B | HAUSSDORF A | HAUSSDORF B |  ASSD A  |  ASSD B  |
|:--------------------:|:------:|:------:|--------|--------|-------------|-------------|----------|----------|
|          0.0         | 0.7333 | 0.7835 | 0.8087 | 0.8561 |   4.4773    |   3.4890    |  1.2458  |  0.9624  |
|          0.05        | 0.7417 | 0.7867 | 0.8158 | 0.8589 |   4.0958    |   3.4073    |  1.1618  |  0.9646  |
|          0.1         | 0.7399 | 0.7827 | 0.8153 | 0.8550 |   4.1999    |   3.4355    |  1.1925  |  0.9735  |
|          0.2         | 0.7421 | 0.7806 | 0.8193 | 0.8522 |   4.2831    |   3.4414    |  1.1953  |  0.9831  |
|          0.3         | 0.7370 | 0.7790 | 0.8134 | 0.8534 |   4.3634    |   3.4972    |  1.2264  |  0.9886  |

## Other
  - Development environment -> CUDA 10.1 and cudnn 7603. Python 3.8.2 - GCC 9.3.0
  - Challenge homepage [here](https://www.ub.edu/mnms/).
  - ACDC nomenclature: 0, 1, 2 and 3 represent voxels located in the background, in the right ventricular cavity, 
    in the myocardium, and in the left ventricular cavity, respectively.

