#!/usr/bin/env bash

# List of checkpoint steps to process
#CHECKPOINTS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350)
eval "$(conda shell.bash hook)"

for epoch in $(seq 1 40); do
  conda activate llama-qw

  # Create temporary config file
  CONTENT="
### model
model_name_or_path: /home/hao/CCFT/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct

### method
stage: sft 
do_train: true 
finetuning_type: lora 
lora_target: all

# deepspeed: /workspace/LLaMA-Factory/configs/deepspeed/ds_z3_config.json  
lora_rank: 256 
lora_alpha: 256 

### dataset
dataset: CCFT_havid_sub_videos_crop_balanced_lh_v0_multi_QA 
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
output_dir: saves/joint_lh_v0_${epoch}
logging_steps: 1 
# save_steps: 50 
plot_loss: true 
#overwrite_output_dir: true 
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
  echo "$CONTENT" > temp_config.yaml

  # Run your command
  echo "Processing epoch ${epoch}..."
  llamafactory-cli train temp_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_config.yaml

  Merge_Adapter="
### model
model_name_or_path: /home/hao/CCFT/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct
adapter_name_or_path: /home/hao/CCFT/LLaMA-Factory/saves/joint_lh_v0_${epoch}
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/hao/CCFT/LLaMA-Factory/models/joint_lh_v0_${epoch}
export_size: 5
export_device: cpu
export_legacy_format: false
"

  # Save config to temporary file
  echo "$Merge_Adapter" > temp_merge_config.yaml

  # Run your command
  echo "Merge lora adapter ${epoch}..."
  llamafactory-cli export temp_merge_config.yaml  # Replace with your actual command

  # Clean up
  rm temp_merge_config.yaml

  conda deactivate

  conda activate Qwen25VL
  python /home/hao/CCFT/Qwen2.5-VL/eval_alternate_single_lh_v0.py --task joint_lh_v0 --epoch ${epoch}
  conda deactivate

done