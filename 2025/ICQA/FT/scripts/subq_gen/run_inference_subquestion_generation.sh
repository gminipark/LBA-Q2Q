CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch --num_processes 8  \
    tasks/subq_gen/inference_subquestion_generation.py \
    --max_source_length 512 \
    --max_target_length 8 \
    --num_proc 4 \
    --q2q_model_path "model_path" \
    --checker_model_path "model_path" \
    --scene_graph_path "./data/train_sceneGraphs.json" \
    --data_dir "./data" \
    --batch_size 1 \
    --num_beams 1 \
    --length_penalty 1.0 \
    --confidence_threshold 0.9 \
    --num_candidate 5 \
    --load_4bit \