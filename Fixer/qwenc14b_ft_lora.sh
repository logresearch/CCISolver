NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
torchrun \
--nproc_per_node 1 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6601 \
../finetune_qwen2.py \
--model_name_or_path "/root/autodl-tmp/Qwen/Qwen2___5-Coder-14B-Instruct" \
--data_path "../data/fixer_merged_diff_conv.json" \
--fp16 True \
--output_dir "../output/qwen2cins_diff_lora_5epoch_enhance" \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 10 \
--learning_rate 1e-4 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 25 \
--report_to "none" \
--model_max_length 1024 \
--gradient_checkpointing True \
--lazy_preprocess True \
--deepspeed "../config/ds_config_zero2.json" \
--use_lora &
