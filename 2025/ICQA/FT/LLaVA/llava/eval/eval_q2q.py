import os
import json
import argparse
import evaluate
from time import time
import pandas as pd

# from nltk.translate.bleu_score import sentence_bleu

def eval_q2q(preds, labels):

    time_eval = time()
    # BLEU 계산
    bleu = evaluate.load("./llava/eval/evaluate/metrics/bleu/bleu.py")
    time_load_bleu = time()
    print("Time taken to load BLEU: ", time_load_bleu - time_eval)
    # BLEU는 각 예측에 대해 하나 이상의 참조가 필요하므로, labels를 리스트의 리스트로 변환
    
    bleu_result = bleu.compute(predictions=preds, references=labels)
    
    # print(bleu_result)
    time_calculate_bleu = time()
    print("Time taken to calculate BLEU: ", time_calculate_bleu - time_load_bleu)

    # ROUGE 계산
    rouge = evaluate.load("./llava/eval/evaluate/metrics/rouge/rouge.py")
    time_load_rouge = time()
    print("Time taken to load ROUGE: ", time_load_rouge - time_calculate_bleu)

    rouge_result = rouge.compute(predictions=preds, references=labels)

    # print(rouge_result)
    time_calculate_rouge = time()
    print("Time taken to calculate ROUGE: ", time_calculate_rouge - time_load_rouge)
    
    # BERTScore 계산
    bertscore = evaluate.load("./llava/eval/evaluate/metrics/bertscore/bertscore.py")
    time_load_bertscore = time()
    print("Time taken to load BERT-SCORE: ", time_load_bertscore - time_calculate_bleu)
    bertscore_result = bertscore.compute(predictions=preds, references=labels, lang="en",device="cuda:0", model_type="distilbert-base-uncased")
    time_calculate_bertscore = time()
    print("Time taken to calculate BLEU: ", time_calculate_bertscore - time_load_bertscore)
    
    # BERTScore는 precision, recall, f1의 리스트를 반환하므로, 평균을 계산
    bertscore_avg = {
        "precision": sum(bertscore_result["precision"]) / len(bertscore_result["precision"]),
        "recall": sum(bertscore_result["recall"]) / len(bertscore_result["recall"]),
        "f1": sum(bertscore_result["f1"]) / len(bertscore_result["f1"]),
    }

    time_bert = time()
    print("Time taken to calculate BERTScore: ", time_bert - time_eval)
    
    
    scores = {
        "BLEU": bleu_result["bleu"],
        "BERTScore": bertscore_avg,
        "ROUGE": rouge_result
    }

    print(scores)
    
    return scores

def get_multi(preds, labels, original_df):
    
    # entity_id별 인덱스 모으기
    entity_to_indices = original_df.groupby('entity_id').apply(lambda df: df.index.tolist())
    
    grouped_preds = []
    grouped_labels = []

    for key, indices in entity_to_indices.items():
        grouped_preds.append(preds[indices[0]])
        grouped_labels.append([labels[i] for i in indices])

    return grouped_preds, grouped_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--result_file", type=str)
    parser.add_argument("--is_multi", action="store_true")
    parser.add_argument("--original_file", type=str, default="")
    args = parser.parse_args()
    
    preds = [json.loads(line)['text'] for line in open(args.result_file, "r")]
    labels = [json.loads(q)["text"] for q in open(args.annotation_file, "r")]
    
    labels = [label.replace("Sub-Question: ","") for label in labels ]
    preds = [pred.replace("Sub-Question: ","") for pred in preds]
    
    if args.is_multi:
        if args.original_file == "":
            base_dir = os.path.dirname(args.annotation_file)
            original_df = pd.read_csv(os.path.join(base_dir, f"Q2Q_test.csv"), dtype={'entity_id': str, 'q_id': str, 'image_id': str})
        else:
            original_df = pd.read_csv(args.original_file, dtype={'entity_id': str, 'q_id': str, 'image_id': str})
        
        original_df = original_df[original_df['label'] == "O"].reset_index()

        preds, labels = get_multi(preds, labels, original_df)
    
    time_start = time()
    
    time_1 = time()
    print("Time taken to load files: ", time_1 - time_start)
    
    metrics = eval_q2q(preds, labels)
    
    with open(args.result_file.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
