deepspeed --num_gpus=0 finetune.py \
    --model_id "Salesforce/blip2-flan-t5-xxl" \
    --dataset_path  "./vqa_aq" \
    --output_dir "output_blip2-flan-t5-xxl_lr1e-4" \
    --overwrite_output_dir True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --max_source_length 128  \
    --max_target_length 32  \
    --learning_rate 1e-4 \
    --evaluation_strategy "epoch" \
    --logging_strategy 'steps' \
    --logging_steps 50 \
    --eval_on_start True \
    --save_strategy "epoch" \
    --prediction_loss_only \
    --bf16 True \
    --fp16 False \
    --dataloader_num_workers 8 \
    --deepspeed "./configs/blip2_t5_z2_config_bf16.json" \