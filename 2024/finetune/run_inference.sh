CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29500 \
    inference.py \
    --dataset_path ./vqa_aq \
    --max_source_length 128 \
    --max_target_length 32 \
    --checkpoint "./output_blip2-flan-t5-xxl_lr3e-4" \
    --num_proc 4 \
    --base_model_name "Salesforce/blip2-flan-t5-xxl" \
    --batch_size 16 \
    --num_beams 4