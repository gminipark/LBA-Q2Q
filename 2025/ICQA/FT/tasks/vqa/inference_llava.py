import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

from PIL import Image

from datasets import load_dataset

from tqdm import tqdm
import argparse
import os
import json 



def collate_fn(examples):
    
    batch = {}
    for key in examples[0].keys():
        batch[key] = torch.tensor([example[key] for example in examples])
        
    return batch



def preprocess_function(examples, 
                        processor,  
                        image_column_name,
                        text_column_name, 
                        target_column_name, 
                        max_source_length, 
                        max_target_length, 
                        prefix=None,
                        padding=False,
                        model_type='llava'):
    
    model_inputs = {}
    images, inputs, targets = [], [], []
    
    
    if model_type == 'llava':
        
        temp_list = []
        for i in range(len(examples[text_column_name])):
            if examples[text_column_name][i] and examples[target_column_name][i]:
                image = Image.open("./data/images/"+str(examples[image_column_name][i])+".jpg").convert('RGB')
                images.append([image])
                
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text":examples[text_column_name][i]
                         + " Answer the question using a single word or phrase."},
                        
                        ],
                    },
                ]
                
                label = examples[target_column_name][i]
                targets.append(label)
                
                text = processor.apply_chat_template(conversation, add_generation_prompt=True)
               
                inputs.append(text)    
                 
        model_inputs = processor(images=images,
                                text=inputs,
                                )
        
    else:
        
        for i in range(len(examples[text_column_name])):
            if examples[text_column_name][i] and examples[target_column_name][i]:
                image = Image.open("./data/images/"+str(examples[image_column_name][i])+".jpg")
                images.append(image)
                
                inputs.append(examples[text_column_name][i])
                label = examples[target_column_name][i]
                targets.append(label)
        
        if prefix:
            inputs = [prefix + " " + "Question: " + inp + " Short answer: " for inp in inputs]
        else:
            inputs = ["Question: " + inp + " Short answer: " for inp in inputs]
        
        model_inputs = processor(images=images,  
                                text=inputs,
                                max_length=max_source_length,
                                padding=padding, 
                                truncation=True,)
    

    labels = processor.tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)    
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

    
def inference(args):
    
    instructions = load_dataset('csv', data_dir='./data', data_files='GQA_Q2Q_test.csv')['train']
    instructions = instructions.filter(lambda example: example['label'] == 'O')
    
    model_id=args.base_model_name

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_id.split("/")[-1] if "/" in model_id else model_id)
    processor.num_additional_image_tokens = 1
    
    model_type = 'llava' if 'llava' in model_id else 'blip'
    
    tokenized_dataset = instructions.map(preprocess_function, batched=True, num_proc=args.num_proc,
                                        fn_kwargs={"processor":  processor, 
                                            'image_column_name': "image_id",
                                            'text_column_name': "ambiguous_question",
                                            'target_column_name': "ambiguous_question_answer",
                                            "max_source_length": args.max_source_length,
                                            "max_target_length": args.max_target_length,
                                            "padding": "max_length",
                                            "prefix": args.prefix,
                                            "model_type": model_type})
    
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    if 'instructblip' in model_id:
        tokenized_dataset = tokenized_dataset.select_columns(['pixel_values', 'qformer_input_ids','input_ids', 'labels'])
    if 'llava' in model_id:
        tokenized_dataset = tokenized_dataset.select_columns(['pixel_values', 'attention_mask','input_ids', 'labels'])    
    else:
        tokenized_dataset = tokenized_dataset.select_columns(['pixel_values', 'input_ids', 'labels'])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    print(tokenized_dataset)
    
    test_dataloader = DataLoader(tokenized_dataset , batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    target_dir_path = './' + model_id.replace("/", "_")
    
    predictions_dir_path = os.path.join(target_dir_path, "predictions")
    if not os.path.exists(predictions_dir_path):
        os.makedirs(predictions_dir_path, exist_ok=True)
     
    
    def accuracy_metric(preds, refs):
        correct = 0
        for pred, ref in zip(preds, refs):
            if pred.lower() == ref.lower():
                correct += 1
        return correct / len(preds)
    
    def compute_metrics(eval_preds):
        preds, refers = eval_preds   
        
        result = {}
        
        result['accuracy'] = accuracy_metric(preds=preds, refs=refers)
        
        return result
        
    model_path_or_name = args.base_model_name
    print(model_path_or_name)
    model = AutoModelForVision2Seq.from_pretrained(model_path_or_name, 
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=model_id.split("/")[-1] if "/" in model_id else model_id,
                                                    torch_dtype=torch.float16,
                                                    device_map="auto")
        
    model.eval()

    predictinos = []
    references = []     
        
    for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            
        with torch.no_grad():
            
            pixel_values = batch['pixel_values'].to(device=model.device)
            input_ids = batch['input_ids'].to(device=model.device)
            labels = batch['labels'].to(device=model.device)
            
            if 'instructblip' in model_id:
                qformer_input_ids = batch['qformer_input_ids'].to(device=model.device)
                outputs = model.generate(pixel_values=pixel_values, qformer_input_ids=qformer_input_ids, input_ids=input_ids, 
                                         max_new_tokens=args.max_target_length, num_beams=args.num_beams, length_penalty=args.length_penalty)
            
            elif 'llava' in model_id:
                
                outputs = model.generate(pixel_values=pixel_values,  input_ids=input_ids, 
                                         max_new_tokens=args.max_target_length, num_beams=args.num_beams, length_penalty=args.length_penalty)
            
            else:
                outputs = model.generate(pixel_values=pixel_values,  input_ids=input_ids, max_new_tokens=args.max_target_length, 
                                         num_beams=args.num_beams, length_penalty=args.length_penalty)

            if 'llava' in model_id:
                decoded_preds = processor.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)
            else:
                decoded_preds = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)    
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
    
        predictinos.extend(decoded_preds)
        references.extend(decoded_labels)
    
    
        
    metrics_rusult = compute_metrics((predictinos, references))

    with open(os.path.join(predictions_dir_path, "metrics_result.txt"), 'w') as f:
        json.dump(metrics_rusult, f)
    
    with open(os.path.join(predictions_dir_path, "predictions.txt"), 'w') as f:
        
        lines = [prediction+'\n' for prediction in predictinos]
        
        f.writelines(lines)
    
    del model
    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=32,
    )
    
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="google/t5-v1_1-large",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
    )
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
    )
    
    args = parser.parse_args()
    
    inference(args)
    

if __name__ == "__main__":
    main()