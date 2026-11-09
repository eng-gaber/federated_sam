#!/bin/bash

CLIENT="C1"

export CUDA_VISIBLE_DEVICES="0"

DATA_ROOT="/home/omanchat-1/Documents/fed/fed_sam/finetune-SAM/polyp_db_dataset/data"
arch="vit_b"
finetune_type="lora"
dataset_name="PolypGen_${CLIENT}"
img_folder="${DATA_ROOT}/data_${CLIENT}"
mask_folder="${DATA_ROOT}/data_${CLIENT}"
train_img_list="${img_folder}/train.csv"
val_img_list="${img_folder}/val.csv"
dir_checkpoint="checkpoints/SAM_${arch}_lora_${dataset_name}"

mkdir -p "$dir_checkpoint"

python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -warmup_period 10 \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -img_folder "$img_folder" \
    -mask_folder "$mask_folder" \
    -sam_ckpt "sam_vit_b_01ec64.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list" \
    -num_cls 2 \
    -b 2 \
    -epochs 2
