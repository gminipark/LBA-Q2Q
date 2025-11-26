SPLIT="test"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_checker \
    --annotation_file ./playground/data/llava_checker_${SPLIT}_bb_answer.jsonl \
    --result_file ./checkpoints/llava-v1.6-13b-checker-lora-2e-4-te-yes/checkpoint-2295/llava_checker_bb_output_${SPLIT}.jsonl \
