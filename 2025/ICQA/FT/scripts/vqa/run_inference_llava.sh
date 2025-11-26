CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29500 \
    tasks/vqa/inference_llava.py \
    --max_source_length 1024 \
    --max_target_length 8 \
    --num_proc 4 \
    --base_model_name "llava-hf/llava-v1.6-vicuna-7b-hf" \
    --batch_size 1 \
    --num_beams 1 \
    --length_penalty 1.0 \
