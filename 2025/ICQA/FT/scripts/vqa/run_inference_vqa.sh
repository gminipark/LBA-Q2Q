CUDA_VISIBLE_DEVICES=0,1 python \
    tasks/vqa/inference_vqa.py \
    --max_source_length 512 \
    --max_target_length 8 \
    --num_proc 4 \
    --base_model_name "llava-hf/llava-v1.6-vicuna-7b-hf" \
    --q2q_model_path "model_path" \
    --answer_model_path "llava-hf/llava-v1.6-vicuna-7b-hf" \
    --ref_path "data/train_sceneGraphs.json" \
    --prediction_dir_name "llava-v1.6-vicuna-7b-auto-direct-beam3" \
    --batch_size 1 \
    --num_beams 3 \
    --length_penalty 1.0 \
    --num_candidate 3 \
    --type "auto-direct" \