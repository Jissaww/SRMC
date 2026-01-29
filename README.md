# **SRDA: Synergy Route Dual-stream Attention Method for Class-agnostic Counting**

## Setup

### Download FSC-147 dataset

Images can be downloaded from here: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing

### Generate density map

if you want to use SRMC, you need to generate the density map.

```
python utils/data.py --data_path <path_to_your_data_directory> --image_size 512 
```

### Weights

few-shot weight MMFA150_3_2shot.pt can download from: https://drive.google.com/file/d/1vozrqA4CNh8Ud49kMFA53odgFFCrp2QD/view?usp=sharing

zero-shot weight MMFA_zero_shot1.pt can download from: https://drive.google.com/file/d/1k468G0SSW6RLC2aqysfICgpQnIX9Y8QD/view?usp=drive_link

CARPK test weight MMFA_3_shot_nocars.pt can download from: https://drive.google.com/file/d/1GBhInGkU3XxhopPgfd2RjaY-mW1Fl2En/view?usp=drive_link

## Training

Please ensure that the relevant paths are modified correctly before training，such as --data_path, -model_path, -log_path, etc.

### Few-shot

```
# multi-GPU：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --pre_norm
# single-GPU:
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --pre_norm
```
### zero-shot

```
# multi-GPU：
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --zero_shot --pre_norm
# single-GPU:
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --zero_shot --pre_norm
```


## Evaluation
### FSC-147

#### few-shot

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 evaluate.py --model_name MMFA150_3_2shot --backbone resnet50 --swav_backbone --reduction 8 --image_size 512 --num_enc_layers 3 --num_ope_iterative_steps 3 --emb_dim 256 --num_heads 8 --kernel_dim 3 --num_objects 3 --pre_norm 
```
#### zero_shot

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 evaluate.py --model_name MMFA_zero_shot1 --backbone resnet50 --swav_backbone --reduction 8 --image_size 512 --num_enc_layers 3 --num_ope_iterative_steps 3 --emb_dim 256 --num_heads 8 --kernel_dim 3 --num_objects 3 --zero_shot --pre_norm 
```

### CARPK

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fsc147_to_carpk.py --pre_norm
```

### COCO

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 fsc147_to_coco.py --batch_size 1 --pre_norm
```

