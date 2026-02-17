# replace the variables with your own, reference this bash scripts

export CUDA_VISIBLE_DEVICES=1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLE=true
export WANDB_MODE=disabled
export WANDB_MODE=offline

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=12347 \
  train/pretrain_unified_navit.py \
  --num_shard 4 \
  --use_lora False \
  # --lora_r 8 \
  # --lora_alpha 32 \
  --visual_gen True \
  --visual_und False \
  --save_every 500 \
  --total_steps 20000 \
  --log_every 1 \
  --warmup_steps 0 \
  --lr 0.0005 \
  --dataset_config_file  \
  --model_path  \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --num_worker 1 \
  --expected_num_tokens 20000 \
  --max_num_tokens 27000 \
  --max_num_tokens_per_sample 27000 \
  --results_dir  \
  --checkpoint_dir  \