SPLIT="test"
MODEL_PATH_DIR="./checkpoints/llava-v1.6-13b-q2q-lora-2e-4-te-yes-b16/checkpoint-582"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_q2q \
    --annotation_file ./playground/data/llava_q2q_${SPLIT}_bb_answer.jsonl \
    --result_file ${MODEL_PATH_DIR}/llava_q2q_bb_output_${SPLIT}.jsonl \
    --is_multi 
