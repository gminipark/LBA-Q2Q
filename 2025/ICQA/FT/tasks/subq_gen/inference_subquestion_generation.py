  
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor,  set_seed
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from datasets import load_dataset, load_from_disk, Value, Features

from tqdm import tqdm
import argparse
import os
import json 
import numpy as np
import pandas as pd

from accelerate import Accelerator, infer_auto_device_map
from accelerate.utils import gather_object, gather
from natsort import natsorted
import random
import queue



def concat_bboxes_to_dataset(instruction, idx, scene_graphs):
        
    imageId = instruction['image_id']
    entityId = instruction['entity_id']
    entity_name = instruction['ambiguous_entity']

    scenegraph = scene_graphs[str(imageId)]
    
    new_info = {"bboxes" : [],
                "labels" : [],
                "target_idx": -1}
    
    target_idx = 0
    for key, value in scenegraph['objects'].items():
        
        if key == entityId:
            new_info['target_idx'] = target_idx
        
        if value['name'] == entity_name:
            new_info['bboxes'].append([value['x'], value['y'], value['x'] + value['w'],value['y'] + value['h']])        
            new_info['labels'].append(entity_name)
            target_idx += 1
            
    instruction.update(new_info)
    
    image = Image.open("./data/images/"+str(instruction['image_id'])+".jpg").convert('RGB')
    
    bounding_boxes = []
    for bbox, label in zip(instruction['bboxes'], instruction['labels']):
        
        x1, y1, x2, y2 = bbox
        width =  image.width
        height = image.height
        x = x1 / width
        y = y1 / height
        x2 = x2 / width
        y2 = y2  / height
        bounding_boxes.append(
            f"{label}: [{x:.3f}, {y:.3f}, {x2:.3f}, {y2:.3f}]"
        )
            
    bounding_boxs_context = ",".join(bounding_boxes)
    
    target_entity_bboxes = bounding_boxes[instruction['target_idx']]
    
    instruction.update({"bounding_boxs_context" : bounding_boxs_context,
                        "target_entity_bboxes" : target_entity_bboxes})
    
           
    
    return instruction

    
def inference(args):
    accelerator = Accelerator()   
    
    target_dir_path = args.data_dir
                
    predictions_dir_path = os.path.join(target_dir_path,f"predictions_conf{args.confidence_threshold}-numc_{args.num_candidate}_0730")
    
    if not os.path.exists(predictions_dir_path):
        os.makedirs(predictions_dir_path, exist_ok=True)

    
    new_sub_questions = []
    undefined_sub_questions = []
    q2q_outputs_list = []

    if os.path.exists(os.path.join(predictions_dir_path, "q2q_outputs.json")):
        accelerator.print('load q2q_outputs.json')
        with accelerator.main_process_first():
            instructions = load_dataset('json', data_files=os.path.join(predictions_dir_path, "q2q_outputs.json"))['train']
            instructions = instructions.map(lambda example, idx: {"original_index": idx}, with_indices=True)
        if accelerator.is_main_process: 
            print(instructions[0])
        
    else:
        # to set sample columns for the dataset
        features = Features({
            'q_id': Value('string'),
            'image_id': Value('string'),
            'entity_id': Value('string'),
            'original_question' : Value('string'),
            'ambiguous_question': Value('string'),
            'ambiguous_question_answer': Value('string'),
            'ambiguous_entity': Value('string'),
        })
        with accelerator.main_process_first():
            instructions = load_dataset('csv', data_dir=args.data_dir, data_files='undefined_Q2Q_sample.csv', features=features)['train']        
            
            instructions = instructions.map(lambda example, idx: {"original_index": idx}, with_indices=True)
            
        if accelerator.is_main_process: 
            print(f"len of instructions: {len(instructions)}")
            print(instructions[0])
            
        if args.scene_graph_path:
            with open(args.scene_graph_path, 'r') as f:
                scene_graphs = json.load(f)
            
            
            instructions = instructions.map(concat_bboxes_to_dataset, 
                                            fn_kwargs={"scene_graphs": scene_graphs }, 
                                            with_indices=True,
                                            load_from_cache_file=False,
                                            )
            print(f"len of instructions: {len(instructions)}")

        # Load the Q2Q model and tokenizer
        if args.q2q_model_path:
            model_name = get_model_name_from_path(args.q2q_model_path)
            q2q_tokenizer, q2q_model, q2q_image_processor, context_len = load_pretrained_model(
                model_path=args.q2q_model_path,
                model_base=None,                        # "liuhaotian/llava-v1.5-13b",
                model_name=model_name,                  # ex) llama-7b-hf
                device_map={"":  accelerator.device},   # {"": accelerator.process_index},
                load_4bit=args.load_4bit
            )
            
            q2q_model.eval()
            
        accelerator.wait_for_everyone()
        
        
        with accelerator.split_between_processes(instructions, apply_padding=True) as split_instructions:
            q2q_pbar = tqdm(split_instructions, total=len(split_instructions) ,desc="Q2Q Inference", disable=not accelerator.is_main_process)
            for instruction in split_instructions:
            # 1. Get bounding boxs context
                image = Image.open("./data/images/"+str(instruction['image_id'])+".jpg").convert('RGB')
                
                bounding_boxs_context = instruction['bounding_boxs_context']
                target_entity_bboxes = instruction['target_entity_bboxes']
                
                # 2. Generate sub-question
                q2q_qs = (
                            bounding_boxs_context + '\n' 
                          + f"Target Entity: {target_entity_bboxes}\n" 
                          + f"Generate a sub-question to classify ambiguous entities. Sub-Question:"
                          )
            
                if q2q_model.config.mm_use_im_start_end:
                    q2q_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q2q_qs
                else:
                    q2q_qs = DEFAULT_IMAGE_TOKEN + '\n' + q2q_qs

                conv = conv_templates['v1'].copy()
                conv.append_message(conv.roles[0], q2q_qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(prompt)
                q2q_input_ids = tokenizer_image_token(prompt, q2q_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(q2q_model.device)

                image_tensor = process_images([image], q2q_image_processor, q2q_model.config)[0]

                with torch.inference_mode():
                    
                    set_seed(42)
                    
                    output_ids = q2q_model.generate(
                        q2q_input_ids,
                        images=image_tensor.unsqueeze(0).half().to(q2q_model.device),
                        image_sizes=[image.size],
                        do_sample=True if args.num_candidate > 1 else False,
                        temperature=0.95 if args.num_candidate > 1 else 1.0,
                        # num_beams=args.num_candidate,
                        num_return_sequences=args.num_candidate,
                        no_repeat_ngram_size=3,
                        max_new_tokens=128,
                        use_cache=True)

                q2q_outputs = q2q_tokenizer.batch_decode(output_ids, skip_special_tokens=True)    
                q2q_outputs = [q2q_output.replace("Sub-Question: ", "").strip() for q2q_output in q2q_outputs]
                
                
                q2q_outputs_list.append((instruction['original_index'], q2q_outputs))
                
                
                accelerator.wait_for_everyone()
                q2q_pbar.update(1)
            
            gather_q2q_outputs = gather_object(q2q_outputs_list)
            
        instruction_dict={}
        for idx, q2q_outputs in gather_q2q_outputs:
            if idx not in instruction_dict.keys():
                instruction_dict[idx]=q2q_outputs 
        
        sorted_gather_q2q_outputs = [instruction_dict[i] for i in range(len(instruction_dict))]
        
        if accelerator.is_main_process:
            print(len(sorted_gather_q2q_outputs))
            print(sorted_gather_q2q_outputs[0])
        
        instructions = instructions.add_column("q2q_outputs", sorted_gather_q2q_outputs)
        
        if accelerator.is_main_process:
            # Save results
            print("Save subquestions")
            instructions.to_json(os.path.join(predictions_dir_path, "q2q_outputs.json"), orient='records', force_ascii=False)
            instructions.to_csv(os.path.join(predictions_dir_path, "q2q_outputs.csv"), index=False)
        
        del q2q_model
    
    accelerator.wait_for_everyone() 
    
     # Checker model
    if args.checker_model_path:
        model_name = get_model_name_from_path(args.checker_model_path)
        checker_tokenizer, checker_model, chekcer_image_processor, context_len = load_pretrained_model(
            model_path=args.checker_model_path,
            model_base=None,
            model_name=model_name,# 예: llama-7b-hf
            device_map={"": accelerator.process_index},
            load_4bit=args.load_4bit
        )
        checker_model.eval()
    
    accelerator.wait_for_everyone()    
    
       
    # Check if the sub-question is correct
    with accelerator.split_between_processes(instructions, apply_padding=True) as split_instructions:            
        
        checker_pbar = tqdm(split_instructions, total=len(split_instructions) ,desc="Checker Inference")
        for idx, instruction in enumerate(split_instructions):
            
            image = Image.open("../images/"+str(instruction['image_id'])+".jpg").convert('RGB')
            
            bounding_boxs_context = instruction['bounding_boxs_context']
            target_entity_bboxes = instruction['target_entity_bboxes']
            
            q2q_outputs = instruction['q2q_outputs']
            # 3. Check if the sub-question is correct
            break_flag = False
            for q2q_output in q2q_outputs:
                sub_question = q2q_output
                
                chekcer_qs = (
                    bounding_boxs_context
                    + f"\n Target Entity: {target_entity_bboxes}"
                    + f"\n Sub-Question: {sub_question}"
                    + f"\n Question: Does the sub-question classify the target entity? Answer:"   
                )
                if checker_model.config.mm_use_im_start_end:
                    chekcer_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + chekcer_qs
                else:
                    chekcer_qs = DEFAULT_IMAGE_TOKEN + '\n' + chekcer_qs

                conv = conv_templates['v1'].copy()
                conv.append_message(conv.roles[0], chekcer_qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()      

                input_ids = tokenizer_image_token(prompt, checker_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(checker_model.device)

                image_tensor = process_images([image], chekcer_image_processor, checker_model.config)[0]

                with torch.inference_mode():
                
                    outputs = checker_model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().to(checker_model.device),
                        image_sizes=[image.size],
                        no_repeat_ngram_size=3,
                        max_new_tokens=128,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True
                        )

                    answer_text = checker_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
                    
                    transition_scores = checker_model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                    confidence = np.exp(transition_scores[0][0].cpu().numpy()) # first token probablity
                
                if 'yes' in answer_text.lower():
                    if args.confidence_threshold > 0:
                        if confidence > args.confidence_threshold:
                            instruction['sub_question'] = sub_question
                            new_sub_questions.append(instruction)
                            break_flag = True
                        
                    else:
                        instruction['sub_question'] = sub_question
                        new_sub_questions.append(instruction)
                        break_flag = True
                    
                if break_flag:
                    break
                
            if break_flag == False:
                instruction['sub_questions'] = q2q_outputs
                undefined_sub_questions.append(instruction)        

            accelerator.wait_for_everyone()            
            checker_pbar.update(1)
    
            
    # Gather all outputs
    gather_new_sub_questions = gather_object(new_sub_questions)
    gather_undefined_sub_questions = gather_object(undefined_sub_questions)
    
    
    # Save results
    if accelerator.is_main_process:
        print("Save alls")
        new_sub_questions_df = pd.DataFrame(gather_new_sub_questions).drop_duplicates(subset=['original_index'])
        new_sub_questions_df.to_csv(os.path.join(predictions_dir_path, "new_sub_questions.csv"), index=False)
        new_sub_questions_df.to_json(os.path.join(predictions_dir_path, "new_sub_questions.json"), index=False, orient='records')
        undefined_sub_questions_df = pd.DataFrame(gather_undefined_sub_questions).drop_duplicates(subset=['original_index'])
        undefined_sub_questions_df.to_csv(os.path.join(predictions_dir_path, "undefined_Q2Q.csv"), index=False)
        undefined_sub_questions_df.to_json(os.path.join(predictions_dir_path, "undefined_Q2Q.json"), index=False, orient='records')

        print("len of toal: ", len(instructions))
        print("len of new_sub_questions: ", len(new_sub_questions_df))
        print("len of undefined_sub_questions: ", len(undefined_sub_questions_df))
    
    del checker_model
    del accelerator

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    
    parser.add_argument(
        "--scene_graph_path",
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
        "--q2q_model_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--checker_model_path",
        type=str,
        default="",
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
    
    parser.add_argument(
        "--num_candidate",
        type=int,
        default=3,
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.0,
    )
    
    
    parser.add_argument(
        "--load_4bit",
        action="store_true",
    )
    
    
    args = parser.parse_args()
    
    inference(args)

    
if __name__ == "__main__":
    main()