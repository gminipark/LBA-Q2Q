import os
import json
import argparse
import evaluate
from time import time

def accuracy(preds, labels):
    return {'accuracy': sum([1 if pred == label else 0 for pred, label in zip(preds, labels)]) / len(preds)}


def eval_q2q(preds, labels):

    time_eval = time()
      # BLEU 계산
    # BLEU는 각 예측에 대해 하나 이상의 참조가 필요하므로, labels를 리스트의 리스트로 변환
    references = [0 if label.lower() == 'yes' else 1 for label in labels]
    predictions = [0 if pred.lower() == 'yes' else 1 for pred in preds]
        
    accuracy_result = accuracy(preds=predictions, labels=references)
    
    time_acc = time()
    print("Time taken to calculate ACC: ", time_acc - time_eval)
    
    return {
        "ACC": accuracy_result["accuracy"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--result_file", type=str)
    args = parser.parse_args()

    time_start = time()
    predictions = [json.loads(line)['text'] for line in open(args.result_file, "r")]
    labels = [json.loads(q)["text"] for q in open(args.annotation_file, "r")]
    
    time_1 = time()
    print("Time taken to load files: ", time_1 - time_start)
    
    metrics = eval_q2q(predictions, labels)
    
    with open(args.result_file.replace(".jsonl", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
