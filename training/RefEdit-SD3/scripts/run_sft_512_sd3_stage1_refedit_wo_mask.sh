#!/bin/bash
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
pretrained_model_name_or_path=BleachNick/SD3_UltraEdit_freeform
ori_model_name_or_path=BleachNick/SD3_UltraEdit_freeform
train_data_jsonl=JSONL_DATA_PATH # for you own dataset
#dataset_name=BleachNick/UltraEdit_500k # 500k for example; better performance using the final 4M data
resolution=512
train_batch_size=8
gradient_accumulation_steps=4
checkpointing_steps=500
validation_step=500

epoch=100
lr=5e-05
output_dir="OUTPUT_DIR" # for you own dataset
accelerate launch --mixed_precision="fp16" --multi_gpu --main_process_port=29555 training/train_sd3_pix2pix_wo_mask.py \
 --pretrained_model_name_or_path ${pretrained_model_name_or_path} \
 --output_dir ${output_dir} \
 --resolution ${resolution} \
 --num_train_epochs ${epoch} \
 --train_batch_size ${train_batch_size} \
 --gradient_accumulation_steps ${gradient_accumulation_steps} \
 --checkpointing_steps ${checkpointing_steps}  \
 --learning_rate ${lr} \
 --lr_warmup_steps 500 \
 --conditioning_dropout_prob 0.05 \
 --mixed_precision fp16 \
 --seed 3345 \
 --num_validation_images 4 \
 --validation_step ${validation_step} \
 --dataloader_num_workers 64 \
 --lr_scheduler cosine \
 --report_to wandb \
 --val_image_url VAL_IMAGE_PATH \
 --validation_prompt "VAL_IMAGE INSTRUCTION" \
 --ori_model_name_or_path ${ori_model_name_or_path} \
 --max_sequence_length 256 \
 --train_data_jsonl ${train_data_jsonl} \
# --dataset_name ${dataset_name} \



