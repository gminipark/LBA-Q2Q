import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

from datasets import load_dataset, load_from_disk

from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from glob import glob
import natsort
import argparse
import os
import evaluate
import json 

from accelerate import Accelerator


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
                        padding=False):
    
    model_inputs = {}
    images, inputs, targets = [], [], []
    for i in range(len(examples[text_column_name])):
        if examples[text_column_name][i] and examples[target_column_name][i]:
            image = Image.open("./ambiguous_images/"+str(examples[image_column_name][i])+".jpg")
            images.append(image)
            
            inputs.append(examples[text_column_name][i])
            targets.append(examples[target_column_name][i])
    
    image_inputs = processor(images=images,  padding="max_length", truncation=True)
    model_inputs.update(image_inputs) 

    if prefix:
        inputs = [prefix + " " + inp for inp in inputs]
    text_inputs = processor.tokenizer(inputs,  
                                      max_length=max_source_length,
                                      padding=padding,
                                      truncation=True
                                    )
    model_inputs['input_ids'] = text_inputs['input_ids']
    
    # Tokenize targets with the `text_target` keyword argument
    labels = processor.tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)    
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

    
def inference(args):
    
    cache_dir = "./data"

    dataset = load_from_disk(args.dataset_path)

    model_id=args.base_model_name

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    
    tokenized_dataset = dataset['test'].map(preprocess_function, batched=True, num_proc=args.num_proc,
                                        fn_kwargs={"processor":  processor, 
                                            'image_column_name': "image_id",
                                            'text_column_name': "ambiguous_question_entity",
                                            'target_column_name': "additional_question",
                                            "max_source_length": args.max_source_length,
                                            "max_target_length": args.max_target_length,
                                            "padding": "max_length"})
    tokenized_dataset = tokenized_dataset.select_columns(['pixel_values', 'input_ids', 'labels'])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")
    print(tokenized_dataset)
    
    test_dataloader = DataLoader(tokenized_dataset , batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    target_dir_path = args.checkpoint
    if args.checkpoint is not None:
        checkpoints = natsort.natsorted(glob(os.path.join(target_dir_path, args.checkpoint_prefix)))

    predictions_dir_path = os.path.join(target_dir_path, "predictions")
    if not os.path.exists(predictions_dir_path):
        os.mkdir(predictions_dir_path)

    accelerator = Accelerator()
    test_dataloader = accelerator.prepare_data_loader(test_dataloader, device_placement=True)
    
    sacrebleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load('bertscore')
    def compute_metrics(eval_preds):
        preds, refers = eval_preds   
        sacrebleu_result = sacrebleu.compute(predictions=preds, references=refers) 
        bertscore_result = bertscore.compute(predictions=preds, references=refers, lang="en", model_type='distillbert-base-uncased')
        
        result = {}
        result['sacrebleu'] = sacrebleu_result
        result['bertscore'] = {'precision_avg': sum(bertscore_result['precision']) / len(bertscore_result['precision']),
                               'recall_avg' : sum(bertscore_result['recall']) / len(bertscore_result['recall']),
                               'f1_avg' : sum(bertscore_result['f1']) / len(bertscore_result['f1'])}
        
        return result
    
    for idx, checkpoint in tqdm(enumerate(checkpoints), total=len(checkpoints), position=0):
        
        
        checkpoint_dir_path = os.path.join(predictions_dir_path, f"checkpoint-{idx+1}")
        if not os.path.exists(checkpoint_dir_path):
            os.mkdir(checkpoint_dir_path)
        else:
            if os.path.exists(os.path.join(checkpoint_dir_path, "predictions.txt")):
                accelerator.free_memory()
                continue
        
        model_path_or_name = checkpoint
        print(model_path_or_name)
        model = AutoModelForVision2Seq.from_pretrained(model_path_or_name, 
                                                       low_cpu_mem_usage=True,
                                                       cache_dir=model_id.split("/")[-1] if "/" in model_id else model_id)
        
        # config = PeftConfig.from_pretrained(checkpoint_dir_path)
        model = PeftModel.from_pretrained(model, checkpoint)

        
        model = accelerator.prepare(model)
        model.eval()

        predictinos = []
        references = []     
        
        samples_seen = 0
        for step, batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader), position=1, disable= not accelerator.is_main_process):
            
            with torch.no_grad():
                
                pixel_values = batch['pixel_values']
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                outputs = accelerator.unwrap_model(model).generate(pixel_values=pixel_values, input_ids=input_ids, max_new_tokens=args.max_target_length, num_beams=args.num_beams)
                #accelerator.print(len(outputs))
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=processor.tokenizer.pad_token_id)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=processor.tokenizer.pad_token_id)

                outputs,labels = accelerator.gather((outputs,labels))
                #accelerator.print(len(outputs))
                
                decoded_preds = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)    
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                if accelerator.num_processes > 1:
                    if step == len(test_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(test_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(test_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)
            
        
            predictinos.extend(decoded_preds)
            references.extend(decoded_labels)
        
        if accelerator.is_main_process:
            
            metrics_rusult = compute_metrics((predictinos, references))

            with open(os.path.join(checkpoint_dir_path, "metrics_result.txt"), 'w') as f:
                json.dump(metrics_rusult, f)
            
            with open(os.path.join(checkpoint_dir_path, "predictions.txt"), 'w') as f:
                
                lines = [prediction+'\n' for prediction in predictinos]
                
                f.writelines(lines)
        
        accelerator.free_memory()
        del model
    
    # accelerator.clear()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    
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
        "--checkpoint",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="checkpoint-*/",
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
    
    args = parser.parse_args()
    
    inference(args)
    

if __name__ == "__main__":
    main()