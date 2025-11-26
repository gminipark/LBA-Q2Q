MODEL_DIR=./checkpoints/model_path
DATA_DIR=./playground/data

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_DIR} \
    --question-file ${DATA_DIR}/llava_checker_test_bb_question.jsonl \
    --image-folder ${DATA_DIR}/ambiguous_images \
    --answers-file ${MODEL_DIR}/llava_checker_bb_output_test_conf.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1