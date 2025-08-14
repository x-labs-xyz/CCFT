#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda activate llama-qw

Objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/best_model/independent/action_verb/rh_v0/action_verb_rh_v0_27 

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
dataset: CCFT_havid_sub_videos_crop_balanced_rh_v0_objects 
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
output_dir: saves/objects_rh_v0_1
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
# overwrite_output_dir: true 
#save_total_limit: 2

### train
per_device_train_batch_size: 1 
gradient_accumulation_steps: 2 
learning_rate: 1.0e-4 
num_train_epochs: 1 
lr_scheduler_type: cosine 
warmup_ratio: 0.1 
bf16: true 
ddp_timeout: 180000000 
  "
# Save config to temporary file
echo "$Objects_CONTENT" > temp_objects_config.yaml
# Run your command
echo "Processing epoch 1 ..."
llamafactory-cli train temp_objects_config.yaml  # Replace with your actual command
# Clean up
rm temp_objects_config.yaml

Merge_objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/best_model/independent/action_verb/rh_v0/action_verb_rh_v0_27 
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/objects_rh_v0_1
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/objects_rh_v0_1
export_size: 5
export_device: cpu
export_legacy_format: false
"
# Save config to temporary file
echo "$Merge_objects_CONTENT" > temp_merge_objects_config.yaml
# Run your command
echo "Merge objects lora adapter 1 ..."
llamafactory-cli export temp_merge_objects_config.yaml  # Replace with your actual command
# Clean up
rm temp_merge_objects_config.yaml

conda deactivate
conda activate Qwen25VL
python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_rh_v0.py --task objects_rh_v0 --epoch "1"
conda deactivate


#Epochs=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# for epoch in {2..20}; do
for epoch in $(seq 2 40); do
  prev_epoch=$((epoch - 1))
  conda activate llama-qw

  Objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/objects_rh_v0_${prev_epoch}

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
dataset: CCFT_havid_sub_videos_crop_balanced_rh_v0_objects 
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
output_dir: saves/objects_rh_v0_${epoch}
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
# overwrite_output_dir: true 
#save_total_limit: 2

### train
per_device_train_batch_size: 1 
gradient_accumulation_steps: 2 
learning_rate: 1.0e-4 
num_train_epochs: 1 
lr_scheduler_type: cosine 
warmup_ratio: 0.1 
bf16: true 
ddp_timeout: 180000000 
  "
  # Save config to temporary file
  echo "$Objects_CONTENT" > temp_objects_config.yaml

  # Run your command
  echo "Processing epoch ${epoch}..."
  llamafactory-cli train temp_objects_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_objects_config.yaml

  Merge_objects_CONTENT="
### model
model_name_or_path: /home/hao/CCFT/LLaMA-Factory/models/objects_rh_v0_${prev_epoch}
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/objects_rh_v0_${epoch}
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/objects_rh_v0_${epoch}
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



  conda deactivate

  conda activate Qwen25VL
  python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_rh_v0.py --task objects_rh_v0 --epoch ${epoch}
  conda deactivate

done