MODEL_DIR=./checkpoints/llava-v1.6-13b-q2q-lora-5e-5-te-yes-b16-e2-steps_e3/checkpoint-2500
DATA_DIR=./playground/data

CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_DIR} \
    --question-file ${DATA_DIR}/llava_q2q_test_bb_question.jsonl \
    --image-folder ${DATA_DIR}/ambiguous_images \
    --answers-file ${MODEL_DIR}/llava_q2q_bb_output_test_b1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_beams 1 \
    # --load_4bit \
    