#!/usr/bin/env bash

# List of checkpoint steps to process
#CHECKPOINTS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350)
eval "$(conda shell.bash hook)"

# conda activate llama-qw

# Action_verb_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct 

# ### method
# stage: sft 
# do_train: true 
# finetuning_type: lora 
# lora_target:
#   - layers.0.self_attn.q_proj
#   - layers.0.self_attn.k_proj
#   - layers.0.self_attn.v_proj
#   - layers.0.self_attn.o_proj
#   - layers.0.mlp.gate_proj
#   - layers.0.mlp.up_proj
#   - layers.0.mlp.down_proj
#   - layers.1.self_attn.q_proj
#   - layers.1.self_attn.k_proj
#   - layers.1.self_attn.v_proj
#   - layers.1.self_attn.o_proj
#   - layers.1.mlp.gate_proj
#   - layers.1.mlp.up_proj
#   - layers.1.mlp.down_proj
#   - layers.2.self_attn.q_proj
#   - layers.2.self_attn.k_proj
#   - layers.2.self_attn.v_proj
#   - layers.2.self_attn.o_proj
#   - layers.2.mlp.gate_proj
#   - layers.2.mlp.up_proj
#   - layers.2.mlp.down_proj
#   - layers.3.self_attn.q_proj
#   - layers.3.self_attn.k_proj
#   - layers.3.self_attn.v_proj
#   - layers.3.self_attn.o_proj
#   - layers.3.mlp.gate_proj
#   - layers.3.mlp.up_proj
#   - layers.3.mlp.down_proj
#   - layers.4.self_attn.q_proj
#   - layers.4.self_attn.k_proj
#   - layers.4.self_attn.v_proj
#   - layers.4.self_attn.o_proj
#   - layers.4.mlp.gate_proj
#   - layers.4.mlp.up_proj
#   - layers.4.mlp.down_proj
#   - layers.5.self_attn.q_proj
#   - layers.5.self_attn.k_proj
#   - layers.5.self_attn.v_proj
#   - layers.5.self_attn.o_proj
#   - layers.5.mlp.gate_proj
#   - layers.5.mlp.up_proj
#   - layers.5.mlp.down_proj
#   - layers.6.self_attn.q_proj
#   - layers.6.self_attn.k_proj
#   - layers.6.self_attn.v_proj
#   - layers.6.self_attn.o_proj
#   - layers.6.mlp.gate_proj
#   - layers.6.mlp.up_proj
#   - layers.6.mlp.down_proj
#   - layers.7.self_attn.q_proj
#   - layers.7.self_attn.k_proj
#   - layers.7.self_attn.v_proj
#   - layers.7.self_attn.o_proj
#   - layers.7.mlp.gate_proj
#   - layers.7.mlp.up_proj
#   - layers.7.mlp.down_pro
#   - layers.8.self_attn.q_proj
#   - layers.8.self_attn.k_proj
#   - layers.8.self_attn.v_proj
#   - layers.8.self_attn.o_proj
#   - layers.8.mlp.gate_proj
#   - layers.8.mlp.up_proj
#   - layers.8.mlp.down_proj
#   - layers.9.self_attn.q_proj
#   - layers.9.self_attn.k_proj
#   - layers.9.self_attn.v_proj
#   - layers.9.self_attn.o_proj
#   - layers.9.mlp.gate_proj
#   - layers.9.mlp.up_proj
#   - layers.9.mlp.down_proj

# # deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
# lora_rank: 256 
# lora_alpha: 256 

# ### dataset
# dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_action_verb 
# template: qwen2_vl 
# cutoff_len: 10240 
# max_samples: 100000 
# overwrite_cache: true 
# preprocessing_num_workers: 16 

# ### Processor Arguments
# image_max_pixels: 589824  #768*768
# image_min_pixels: 102400 #320*320
# video_max_pixels: 102400 #360*360
# video_min_pixels: 102400 #320*320
# video_fps: 2
# video_maxlen: 128

# ### output
# output_dir: saves/action_verb_lh_v1_1
# logging_steps: 1 
# #save_steps: 50 
# plot_loss: true 
# #overwrite_output_dir: true 
# #save_total_limit: 2

# ### train
# per_device_train_batch_size: 1 
# gradient_accumulation_steps: 2 
# learning_rate: 1.0e-4 
# num_train_epochs: 1
# lr_scheduler_type: cosine 
# warmup_ratio: 0.1 
# bf16: true  
# ddp_timeout: 180000000 
# "

# # Save config to temporary file
# echo "$Action_verb_CONTENT" > temp_action_verb_config.yaml
# # Run your command
# echo "Processing epoch 1 ..."
# llamafactory-cli train temp_action_verb_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_action_verb_config.yaml

# Merge_action_verb_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct
# adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_lh_v1_1
# template: qwen2_vl
# finetuning_type: lora

# ### export
# export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_1
# export_size: 5
# export_device: cpu
# export_legacy_format: false
# "
# # Save config to temporary file
# echo "$Merge_action_verb_CONTENT" > temp_merge_action_verb_config.yaml
# # Run your command
# echo "Merge action verb lora adapter 1 ..."
# llamafactory-cli export temp_merge_action_verb_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_merge_action_verb_config.yaml

# Objects_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_1

# ### method
# stage: sft 
# do_train: true 
# finetuning_type: lora 
# lora_target:
#   - layers.10.self_attn.q_proj
#   - layers.10.self_attn.k_proj
#   - layers.10.self_attn.v_proj
#   - layers.10.self_attn.o_proj
#   - layers.10.mlp.gate_proj
#   - layers.10.mlp.up_proj
#   - layers.10.mlp.down_proj
#   - layers.11.self_attn.q_proj
#   - layers.11.self_attn.k_proj
#   - layers.11.self_attn.v_proj
#   - layers.11.self_attn.o_proj
#   - layers.11.mlp.gate_proj
#   - layers.11.mlp.up_proj
#   - layers.11.mlp.down_proj
#   - layers.12.self_attn.q_proj
#   - layers.12.self_attn.k_proj
#   - layers.12.self_attn.v_proj
#   - layers.12.self_attn.o_proj
#   - layers.12.mlp.gate_proj
#   - layers.12.mlp.up_proj
#   - layers.12.mlp.down_proj
#   - layers.13.self_attn.q_proj
#   - layers.13.self_attn.k_proj
#   - layers.13.self_attn.v_proj
#   - layers.13.self_attn.o_proj
#   - layers.13.mlp.gate_proj
#   - layers.13.mlp.up_proj
#   - layers.13.mlp.down_proj
#   - layers.14.self_attn.q_proj
#   - layers.14.self_attn.k_proj
#   - layers.14.self_attn.v_proj
#   - layers.14.self_attn.o_proj
#   - layers.14.mlp.gate_proj
#   - layers.14.mlp.up_proj
#   - layers.14.mlp.down_proj
#   - layers.15.self_attn.q_proj
#   - layers.15.self_attn.k_proj
#   - layers.15.self_attn.v_proj
#   - layers.15.self_attn.o_proj
#   - layers.15.mlp.gate_proj
#   - layers.15.mlp.up_proj
#   - layers.15.mlp.down_proj
#   - layers.16.self_attn.q_proj
#   - layers.16.self_attn.k_proj
#   - layers.16.self_attn.v_proj
#   - layers.16.self_attn.o_proj
#   - layers.16.mlp.gate_proj
#   - layers.16.mlp.up_proj
#   - layers.16.mlp.down_proj
#   - layers.17.self_attn.q_proj
#   - layers.17.self_attn.k_proj
#   - layers.17.self_attn.v_proj
#   - layers.17.self_attn.o_proj
#   - layers.17.mlp.gate_proj
#   - layers.17.mlp.up_proj
#   - layers.17.mlp.down_proj
#   - layers.18.self_attn.q_proj
#   - layers.18.self_attn.k_proj
#   - layers.18.self_attn.v_proj
#   - layers.18.self_attn.o_proj
#   - layers.18.mlp.gate_proj
#   - layers.18.mlp.up_proj
#   - layers.18.mlp.down_proj
#   - layers.19.self_attn.q_proj
#   - layers.19.self_attn.k_proj
#   - layers.19.self_attn.v_proj
#   - layers.19.self_attn.o_proj
#   - layers.19.mlp.gate_proj
#   - layers.19.mlp.up_proj
#   - layers.19.mlp.down_proj
#   - layers.20.self_attn.q_proj
#   - layers.20.self_attn.k_proj
#   - layers.20.self_attn.v_proj
#   - layers.20.self_attn.o_proj
#   - layers.20.mlp.gate_proj
#   - layers.20.mlp.up_proj
#   - layers.20.mlp.down_proj
#   - layers.21.self_attn.q_proj
#   - layers.21.self_attn.k_proj
#   - layers.21.self_attn.v_proj
#   - layers.21.self_attn.o_proj
#   - layers.21.mlp.gate_proj
#   - layers.21.mlp.up_proj
#   - layers.21.mlp.down_proj
#   - layers.22.self_attn.q_proj
#   - layers.22.self_attn.k_proj
#   - layers.22.self_attn.v_proj
#   - layers.22.self_attn.o_proj
#   - layers.22.mlp.gate_proj
#   - layers.22.mlp.up_proj
#   - layers.22.mlp.down_proj
#   - layers.23.self_attn.q_proj
#   - layers.23.self_attn.k_proj
#   - layers.23.self_attn.v_proj
#   - layers.23.self_attn.o_proj
#   - layers.23.mlp.gate_proj
#   - layers.23.mlp.up_proj
#   - layers.23.mlp.down_proj

# # deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
# lora_rank: 256 
# lora_alpha: 256 


# ### dataset
# dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_objects 
# template: qwen2_vl 
# cutoff_len: 10240 
# max_samples: 100000 
# overwrite_cache: true 
# preprocessing_num_workers: 16 

# ### Processor Arguments
# image_max_pixels: 589824  #768*768
# image_min_pixels: 102400 #320*320
# video_max_pixels: 102400 #360*360
# video_min_pixels: 102400 #320*320
# video_fps: 2
# video_maxlen: 128

# ### output
# output_dir: saves/action_verb_objects_lh_v1_1
# logging_steps: 1 
# # save_steps: 50 
# plot_loss: true 
# # overwrite_output_dir: true 
# #save_total_limit: 2

# ### train
# per_device_train_batch_size: 1 
# gradient_accumulation_steps: 2 
# learning_rate: 1.0e-4 
# num_train_epochs: 1 
# lr_scheduler_type: cosine 
# warmup_ratio: 0.1 
# bf16: true 
# ddp_timeout: 180000000 
#   "
# # Save config to temporary file
# echo "$Objects_CONTENT" > temp_action_verb_objects_config.yaml
# # Run your command
# echo "Processing epoch 1 ..."
# llamafactory-cli train temp_action_verb_objects_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_action_verb_objects_config.yaml

# Merge_objects_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_1
# adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_objects_lh_v1_1
# template: qwen2_vl
# finetuning_type: lora

# ### export
# export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_1
# export_size: 5
# export_device: cpu
# export_legacy_format: false
# "
# # Save config to temporary file
# echo "$Merge_objects_CONTENT" > temp_merge_objects_config.yaml
# # Run your command
# echo "Merge objects lora adapter 1 ..."
# llamafactory-cli export temp_merge_objects_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_merge_objects_config.yaml

# Tool_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_1

# ### method
# stage: sft 
# do_train: true 
# finetuning_type: lora 
# lora_target:
#   - layers.24.self_attn.q_proj
#   - layers.24.self_attn.k_proj
#   - layers.24.self_attn.v_proj
#   - layers.24.self_attn.o_proj
#   - layers.24.mlp.gate_proj
#   - layers.24.mlp.up_proj
#   - layers.24.mlp.down_proj
#   - layers.25.self_attn.q_proj
#   - layers.25.self_attn.k_proj
#   - layers.25.self_attn.v_proj
#   - layers.25.self_attn.o_proj
#   - layers.25.mlp.gate_proj
#   - layers.25.mlp.up_proj
#   - layers.25.mlp.down_proj
#   - layers.26.self_attn.q_proj
#   - layers.26.self_attn.k_proj
#   - layers.26.self_attn.v_proj
#   - layers.26.self_attn.o_proj
#   - layers.26.mlp.gate_proj
#   - layers.26.mlp.up_proj
#   - layers.26.mlp.down_proj
#   - layers.27.self_attn.q_proj
#   - layers.27.self_attn.k_proj
#   - layers.27.self_attn.v_proj
#   - layers.27.self_attn.o_proj
#   - layers.27.mlp.gate_proj
#   - layers.27.mlp.up_proj
#   - layers.27.mlp.down_proj

# # deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
# lora_rank: 256 
# lora_alpha: 256 

# ### dataset
# dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_tool 
# template: qwen2_vl 
# cutoff_len: 10240 
# max_samples: 100000 
# overwrite_cache: true 
# preprocessing_num_workers: 16 

# ### Processor Arguments
# image_max_pixels: 589824  #768*768
# image_min_pixels: 102400 #320*320
# video_max_pixels: 102400 #768*768
# video_min_pixels: 102400 #320*320
# video_fps: 2
# video_maxlen: 128

# ### output
# output_dir: saves/action_verb_objects_tool_lh_v1_1
# logging_steps: 1 
# # save_steps: 50 
# plot_loss: true 
# # overwrite_output_dir: true 
# #save_total_limit: 2

# ### train
# per_device_train_batch_size: 1 
# gradient_accumulation_steps: 2 
# learning_rate: 1.0e-4 
# num_train_epochs: 1 
# lr_scheduler_type: cosine
# warmup_ratio: 0.1 
# bf16: true  
# ddp_timeout: 180000000 
#   "
# # Save config to temporary file
# echo "$Tool_CONTENT" > temp_action_verb_objects_tool_config.yaml
# # Run your command
# echo "Processing epoch 1..."
# llamafactory-cli train temp_action_verb_objects_tool_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_action_verb_objects_tool_config.yaml

# Merge_tool_CONTENT="
# ### model
# model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_1
# adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_objects_tool_lh_v1_1
# template: qwen2_vl
# finetuning_type: lora

# ### export
# export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_tool_lh_v1_1
# export_size: 5
# export_device: cpu
# export_legacy_format: false
# "
# # Save config to temporary file
# echo "$Merge_tool_CONTENT" > temp_merge_tool_config.yaml
# # Run your command
# echo "Merge tool lora adapter 1 ..."
# llamafactory-cli export temp_merge_tool_config.yaml  # Replace with your actual command
# # Clean up
# rm temp_merge_tool_config.yaml

# conda deactivate
# conda activate Qwen25VL
# python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_objects_tool_lh_v1 --epoch "1"
# python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_objects_lh_v1 --epoch "1"
# python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_lh_v1 --epoch "1"
# conda deactivate

#Epochs=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# for epoch in {2..20}; do
for epoch in $(seq 19 40); do
  prev_epoch=$((epoch - 1))
  conda activate llama-qw

  # Create temporary config file
  Action_verb_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_tool_lh_v1_${prev_epoch}

### method
stage: sft 
do_train: true 
finetuning_type: lora 
lora_target:
  - layers.0.self_attn.q_proj
  - layers.0.self_attn.k_proj
  - layers.0.self_attn.v_proj
  - layers.0.self_attn.o_proj
  - layers.0.mlp.gate_proj
  - layers.0.mlp.up_proj
  - layers.0.mlp.down_proj
  - layers.1.self_attn.q_proj
  - layers.1.self_attn.k_proj
  - layers.1.self_attn.v_proj
  - layers.1.self_attn.o_proj
  - layers.1.mlp.gate_proj
  - layers.1.mlp.up_proj
  - layers.1.mlp.down_proj
  - layers.2.self_attn.q_proj
  - layers.2.self_attn.k_proj
  - layers.2.self_attn.v_proj
  - layers.2.self_attn.o_proj
  - layers.2.mlp.gate_proj
  - layers.2.mlp.up_proj
  - layers.2.mlp.down_proj
  - layers.3.self_attn.q_proj
  - layers.3.self_attn.k_proj
  - layers.3.self_attn.v_proj
  - layers.3.self_attn.o_proj
  - layers.3.mlp.gate_proj
  - layers.3.mlp.up_proj
  - layers.3.mlp.down_proj
  - layers.4.self_attn.q_proj
  - layers.4.self_attn.k_proj
  - layers.4.self_attn.v_proj
  - layers.4.self_attn.o_proj
  - layers.4.mlp.gate_proj
  - layers.4.mlp.up_proj
  - layers.4.mlp.down_proj
  - layers.5.self_attn.q_proj
  - layers.5.self_attn.k_proj
  - layers.5.self_attn.v_proj
  - layers.5.self_attn.o_proj
  - layers.5.mlp.gate_proj
  - layers.5.mlp.up_proj
  - layers.5.mlp.down_proj
  - layers.6.self_attn.q_proj
  - layers.6.self_attn.k_proj
  - layers.6.self_attn.v_proj
  - layers.6.self_attn.o_proj
  - layers.6.mlp.gate_proj
  - layers.6.mlp.up_proj
  - layers.6.mlp.down_proj
  - layers.7.self_attn.q_proj
  - layers.7.self_attn.k_proj
  - layers.7.self_attn.v_proj
  - layers.7.self_attn.o_proj
  - layers.7.mlp.gate_proj
  - layers.7.mlp.up_proj
  - layers.7.mlp.down_pro
  - layers.8.self_attn.q_proj
  - layers.8.self_attn.k_proj
  - layers.8.self_attn.v_proj
  - layers.8.self_attn.o_proj
  - layers.8.mlp.gate_proj
  - layers.8.mlp.up_proj
  - layers.8.mlp.down_proj
  - layers.9.self_attn.q_proj
  - layers.9.self_attn.k_proj
  - layers.9.self_attn.v_proj
  - layers.9.self_attn.o_proj
  - layers.9.mlp.gate_proj
  - layers.9.mlp.up_proj
  - layers.9.mlp.down_proj

# deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
lora_rank: 256 
lora_alpha: 256 

### dataset
dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_action_verb 
template: qwen2_vl 
cutoff_len: 10240 
max_samples: 100000 
overwrite_cache: true 
preprocessing_num_workers: 16 

### Processor Arguments
image_max_pixels: 589824  #768*768
image_min_pixels: 102400 #320*320
video_max_pixels: 102400 #360*360
video_min_pixels: 102400 #320*320
video_fps: 2
video_maxlen: 128

### output
output_dir: saves/action_verb_lh_v1_${epoch}
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
#overwrite_output_dir: true 
#save_total_limit: 2

### train
per_device_train_batch_size: 1 
gradient_accumulation_steps: 2 
learning_rate: 5.0e-5 
num_train_epochs: 1
lr_scheduler_type: cosine 
warmup_ratio: 0.1 
bf16: true  
ddp_timeout: 180000000 
"

  # Save config to temporary file
  echo "$Action_verb_CONTENT" > temp_action_verb_config.yaml

  # Run your command
  echo "Processing epoch ${epoch}..."
  llamafactory-cli train temp_action_verb_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_action_verb_config.yaml

  Merge_action_verb_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_tool_lh_v1_${prev_epoch}
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_lh_v1_${epoch}
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_${epoch}
export_size: 5
export_device: cpu
export_legacy_format: false
"

  # Save config to temporary file
  echo "$Merge_action_verb_CONTENT" > temp_merge_action_verb_config.yaml

  # Run your command
  echo "Merge action verb lora adapter ${epoch}..."
  llamafactory-cli export temp_merge_action_verb_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_merge_action_verb_config.yaml

  Objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_${epoch}

### method
stage: sft 
do_train: true 
finetuning_type: lora 
lora_target:
  - layers.10.self_attn.q_proj
  - layers.10.self_attn.k_proj
  - layers.10.self_attn.v_proj
  - layers.10.self_attn.o_proj
  - layers.10.mlp.gate_proj
  - layers.10.mlp.up_proj
  - layers.10.mlp.down_proj
  - layers.11.self_attn.q_proj
  - layers.11.self_attn.k_proj
  - layers.11.self_attn.v_proj
  - layers.11.self_attn.o_proj
  - layers.11.mlp.gate_proj
  - layers.11.mlp.up_proj
  - layers.11.mlp.down_proj
  - layers.12.self_attn.q_proj
  - layers.12.self_attn.k_proj
  - layers.12.self_attn.v_proj
  - layers.12.self_attn.o_proj
  - layers.12.mlp.gate_proj
  - layers.12.mlp.up_proj
  - layers.12.mlp.down_proj
  - layers.13.self_attn.q_proj
  - layers.13.self_attn.k_proj
  - layers.13.self_attn.v_proj
  - layers.13.self_attn.o_proj
  - layers.13.mlp.gate_proj
  - layers.13.mlp.up_proj
  - layers.13.mlp.down_proj
  - layers.14.self_attn.q_proj
  - layers.14.self_attn.k_proj
  - layers.14.self_attn.v_proj
  - layers.14.self_attn.o_proj
  - layers.14.mlp.gate_proj
  - layers.14.mlp.up_proj
  - layers.14.mlp.down_proj
  - layers.15.self_attn.q_proj
  - layers.15.self_attn.k_proj
  - layers.15.self_attn.v_proj
  - layers.15.self_attn.o_proj
  - layers.15.mlp.gate_proj
  - layers.15.mlp.up_proj
  - layers.15.mlp.down_proj
  - layers.16.self_attn.q_proj
  - layers.16.self_attn.k_proj
  - layers.16.self_attn.v_proj
  - layers.16.self_attn.o_proj
  - layers.16.mlp.gate_proj
  - layers.16.mlp.up_proj
  - layers.16.mlp.down_proj
  - layers.17.self_attn.q_proj
  - layers.17.self_attn.k_proj
  - layers.17.self_attn.v_proj
  - layers.17.self_attn.o_proj
  - layers.17.mlp.gate_proj
  - layers.17.mlp.up_proj
  - layers.17.mlp.down_proj
  - layers.18.self_attn.q_proj
  - layers.18.self_attn.k_proj
  - layers.18.self_attn.v_proj
  - layers.18.self_attn.o_proj
  - layers.18.mlp.gate_proj
  - layers.18.mlp.up_proj
  - layers.18.mlp.down_proj
  - layers.19.self_attn.q_proj
  - layers.19.self_attn.k_proj
  - layers.19.self_attn.v_proj
  - layers.19.self_attn.o_proj
  - layers.19.mlp.gate_proj
  - layers.19.mlp.up_proj
  - layers.19.mlp.down_proj
  - layers.20.self_attn.q_proj
  - layers.20.self_attn.k_proj
  - layers.20.self_attn.v_proj
  - layers.20.self_attn.o_proj
  - layers.20.mlp.gate_proj
  - layers.20.mlp.up_proj
  - layers.20.mlp.down_proj
  - layers.21.self_attn.q_proj
  - layers.21.self_attn.k_proj
  - layers.21.self_attn.v_proj
  - layers.21.self_attn.o_proj
  - layers.21.mlp.gate_proj
  - layers.21.mlp.up_proj
  - layers.21.mlp.down_proj
  - layers.22.self_attn.q_proj
  - layers.22.self_attn.k_proj
  - layers.22.self_attn.v_proj
  - layers.22.self_attn.o_proj
  - layers.22.mlp.gate_proj
  - layers.22.mlp.up_proj
  - layers.22.mlp.down_proj
  - layers.23.self_attn.q_proj
  - layers.23.self_attn.k_proj
  - layers.23.self_attn.v_proj
  - layers.23.self_attn.o_proj
  - layers.23.mlp.gate_proj
  - layers.23.mlp.up_proj
  - layers.23.mlp.down_proj

# deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
lora_rank: 256 
lora_alpha: 256 


### dataset
dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_objects 
template: qwen2_vl 
cutoff_len: 10240 
max_samples: 100000 
overwrite_cache: true 
preprocessing_num_workers: 16 

### Processor Arguments
image_max_pixels: 589824  #768*768
image_min_pixels: 102400 #320*320
video_max_pixels: 102400 #360*360
video_min_pixels: 102400 #320*320
video_fps: 2
video_maxlen: 128

### output
output_dir: saves/action_verb_objects_lh_v1_${epoch}
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
# overwrite_output_dir: true 
#save_total_limit: 2

### train
per_device_train_batch_size: 1 
gradient_accumulation_steps: 2 
learning_rate: 5.0e-5 
num_train_epochs: 1 
lr_scheduler_type: cosine 
warmup_ratio: 0.1 
bf16: true 
ddp_timeout: 180000000 
  "
  # Save config to temporary file
  echo "$Objects_CONTENT" > temp_action_verb_objects_config.yaml

  # Run your command
  echo "Processing epoch ${epoch}..."
  llamafactory-cli train temp_action_verb_objects_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_action_verb_objects_config.yaml

  Merge_objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_lh_v1_${epoch}
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_objects_lh_v1_${epoch}
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_${epoch}
export_size: 5
export_device: cpu
export_legacy_format: false
"

  # Save config to temporary file
  echo "$Merge_objects_CONTENT" > temp_merge_objects_config.yaml

  # Run your command
  echo "Merge objects lora adapter ${epoch}..."
  llamafactory-cli export temp_merge_objects_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_merge_objects_config.yaml

  Tool_CONTENT="
  ### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_${epoch}

### method
stage: sft 
do_train: true 
finetuning_type: lora 
lora_target:
  - layers.24.self_attn.q_proj
  - layers.24.self_attn.k_proj
  - layers.24.self_attn.v_proj
  - layers.24.self_attn.o_proj
  - layers.24.mlp.gate_proj
  - layers.24.mlp.up_proj
  - layers.24.mlp.down_proj
  - layers.25.self_attn.q_proj
  - layers.25.self_attn.k_proj
  - layers.25.self_attn.v_proj
  - layers.25.self_attn.o_proj
  - layers.25.mlp.gate_proj
  - layers.25.mlp.up_proj
  - layers.25.mlp.down_proj
  - layers.26.self_attn.q_proj
  - layers.26.self_attn.k_proj
  - layers.26.self_attn.v_proj
  - layers.26.self_attn.o_proj
  - layers.26.mlp.gate_proj
  - layers.26.mlp.up_proj
  - layers.26.mlp.down_proj
  - layers.27.self_attn.q_proj
  - layers.27.self_attn.k_proj
  - layers.27.self_attn.v_proj
  - layers.27.self_attn.o_proj
  - layers.27.mlp.gate_proj
  - layers.27.mlp.up_proj
  - layers.27.mlp.down_proj

# deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
lora_rank: 256 
lora_alpha: 256 

### dataset
dataset: CCFT_havid_sub_videos_crop_balanced_lh_v1_tool 
template: qwen2_vl 
cutoff_len: 10240 
max_samples: 100000 
overwrite_cache: true 
preprocessing_num_workers: 16 

### Processor Arguments
image_max_pixels: 589824  #768*768
image_min_pixels: 102400 #320*320
video_max_pixels: 102400 #768*768
video_min_pixels: 102400 #320*320
video_fps: 2
video_maxlen: 128

### output
output_dir: saves/action_verb_objects_tool_lh_v1_${epoch}
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
# overwrite_output_dir: true 
#save_total_limit: 2

### train
per_device_train_batch_size: 1 
gradient_accumulation_steps: 2 
learning_rate: 5.0e-5 
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.1 
bf16: true  
ddp_timeout: 180000000 
  "
  # Save config to temporary file
  echo "$Tool_CONTENT" > temp_action_verb_objects_tool_config.yaml

  # Run your command
  echo "Processing epoch ${epoch}..."
  llamafactory-cli train temp_action_verb_objects_tool_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_action_verb_objects_tool_config.yaml

  Merge_tool_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_lh_v1_${epoch}
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/action_verb_objects_tool_lh_v1_${epoch}
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/action_verb_objects_tool_lh_v1_${epoch}
export_size: 5
export_device: cpu
export_legacy_format: false
"

  # Save config to temporary file
  echo "$Merge_tool_CONTENT" > temp_merge_tool_config.yaml

  # Run your command
  echo "Merge tool lora adapter ${epoch}..."
  llamafactory-cli export temp_merge_tool_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_merge_tool_config.yaml

  conda deactivate

  conda activate Qwen25VL
  python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_objects_tool_lh_v1 --epoch ${epoch}
  python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_objects_lh_v1 --epoch ${epoch}
  python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v1.py --task action_verb_lh_v1 --epoch ${epoch}
  conda deactivate

done