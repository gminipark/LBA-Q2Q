  
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, set_seed
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from datasets import load_dataset, Value, Features

from tqdm import tqdm
import argparse
import os
import json 
import numpy as np


def concat_ref_to_dataset(instruction, idx,refs):
        
    imageId = instruction['image_id']
    entityId = instruction['entity_id']
    entity_name = instruction['ambiguous_entity']

    scenegraph = refs[str(imageId)]
    
    new_info = {"bboxes" : [],
                "labels" : [],
                "target_idx": -1}
    
    idx = 0
    target_entity_name = scenegraph['objects'][entityId]['name']
    for key, value in scenegraph['objects'].items():
        
        if str(key) == str(entityId):
            new_info['target_idx'] = idx
            
    
        if value['name'] == target_entity_name:
            new_info['bboxes'].append([value['x'], value['y'], value['x'] + value['w'],value['y'] + value['h']])        
            new_info['labels'].append(entity_name)
            idx += 1

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
    try:
        target_entity_bboxes = bounding_boxes[instruction['target_idx']]
    except:
        print(entityId)
        print(scenegraph['objects'].keys())
        print(instruction['target_idx'])
        print(bounding_boxes)
        print([obj['name'] for obj in scenegraph['objects'].values()])
        print(entity_name)
        print(instruction['original_question'])
        print(instruction['ambiguous_question'])
        
    instruction.update({'bounding_boxes' : bounding_boxes,
                        "bounding_boxes_context" : bounding_boxs_context,
                        "target_entity_bboxes" : target_entity_bboxes})
    
    
    return instruction


        
    
def inference(args):
    
    
    features = Features({
            'q_id': Value('string'),
            'image_id': Value('string'),
            'entity_id': Value('string'),
            'original_question' : Value('string'),
            'ambiguous_question': Value('string'),
            'ambiguous_question_answer': Value('string'),
            'ambiguous_entity': Value('string'),
            'additional_question': Value('string'),
            'label': Value('string'),
    })
    
    instructions = load_dataset('csv', data_dir='./data', data_files='GQA_Q2Q_test.csv', features=features)['train']

    
    instructions = instructions.filter(lambda example: example['label'] == 'O')
    
    if args.ref_path:
        with open(args.ref_path, 'r') as f:
            refs = json.load(f)
        
        instructions = instructions.map(concat_ref_to_dataset, fn_kwargs={"refs": refs }, with_indices=True,load_from_cache_file=False)
        print(instructions[0])
    
    model_id=args.base_model_name


    processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_id.split("/")[-1] if "/" in model_id else model_id)
    
    target_dir_path = './' + model_id.replace("/", "_")
    
    predictions_dir_path = os.path.join(target_dir_path, args.prediction_dir_name ,"predictions")
    if not os.path.exists(predictions_dir_path):
        os.makedirs(predictions_dir_path, exist_ok=True)
     
    
    def accuracy_metric(preds, refs):
        correct = 0
        for pred, ref in zip(preds, refs):
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
        return correct / len(preds)
    
    def compute_metrics(eval_preds):
        preds, refers = eval_preds   
        
        result = {}
        
        result['accuracy'] = accuracy_metric(preds=preds, refs=refers)
        
        return result
        
    model_path_or_name = args.base_model_name
  
    model = AutoModelForVision2Seq.from_pretrained(model_path_or_name, 
                                                low_cpu_mem_usage=True,
                                                cache_dir=os.path.join('models',model_id.split("/")[-1] if "/" in model_id else model_id),
                                                torch_dtype=torch.float16,
                                                device_map="auto",
                                                )

    
    model.eval()
    if args.q2q_model_path:
        model_name = get_model_name_from_path(args.q2q_model_path)
        q2q_tokenizer, q2q_model, q2q_image_processor, context_len = load_pretrained_model(
            model_path=args.q2q_model_path,
            model_base=None, #liuhaotian/llava-v1.5-13b
            model_name=model_name,# 예: llama-7b-hf
            device_map='auto',
            # load_4bit=True,
        )
        q2q_model.eval()
       
    if args.answer_model_path:
        
        if args.answer_model_path == model_path_or_name:
            answer_model = model
            answer_processor = processor
        else:
            answer_model = AutoModelForVision2Seq.from_pretrained(args.answer_model_path, 
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=os.path.join('models',model_id.split("/")[-1] if "/" in model_id else model_id),
                                                    torch_dtype=torch.float16,
                                                    device_map='auto')
            answer_processor = AutoProcessor.from_pretrained(args.answer_model_path)
            answer_model.eval()
    
    if args.checker_model_path:
        model_name = get_model_name_from_path(args.checker_model_path)
        checker_tokenizer, checker_model, checker_image_processor, context_len = load_pretrained_model(
            model_path=args.q2q_model_path,
            model_base=None, #liuhaotian/llava-v1.5-13b
            model_name=model_name,# 예: llama-7b-hf
            device_map='auto',
            # load_4bit=True,
        )
        checker_model.eval()
        

    predictions = []
    references = []     
    input_list = []
        
    for instruction in tqdm(instructions, total=len(instructions['original_question'])):
            
        with torch.no_grad():
            image = Image.open("./data/images/"+str(instruction['image_id'])+".jpg").convert('RGB')
                
            question = instruction['ambiguous_question']
            answer = instruction['ambiguous_question_answer']
            
            bounding_boxs_context = instruction['bounding_boxes_context']
            bounding_boxes = instruction['bounding_boxes']
            
            if "test" in args.type:
                sub_question = instruction['additional_question']
                if 'machine' in args.type:
                    answer_conversation = [
                            {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": (bounding_boxs_context 
                                                        + '\n' 
                                                        + f"Target Entity: {bounding_boxes[instruction['target_idx']]}"
                                                        + "\n" + '\n' + sub_question \
                                                        + " Answer the question using Yes or No.")},
                                
                                ],
                            }
                        ]
                    answer_text = answer_processor.apply_chat_template(answer_conversation, add_generation_prompt=True)
                    
                    answer_inputs = answer_processor(text=answer_text, images=image, return_tensors='pt').to(answer_model.device)
                    answer_outputs = answer_model.generate(**answer_inputs, max_new_tokens=16)
                    
                    answer_decoded_preds = answer_processor.tokenizer.batch_decode(answer_outputs[:, answer_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    answer_text = answer_decoded_preds[0].strip()
                    
                    if 'yes' in answer_text.lower():
                        answer_text = "Yes"
                    else:
                        answer_text = "No"
                else:
                    answer_text = 'Yes'
                    
                # if answer_text == "Yes":
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{sub_question}"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"{answer_text}"},
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question + " Answer the question using a single word or phrase."},
                        ]
                    }
                ]
                
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=prompt, images=image,return_tensors='pt').to(model.device)
                
                outputs = model.generate(**inputs,  max_new_tokens=args.max_target_length, 
                                            num_beams=args.num_beams, length_penalty=args.length_penalty, do_sample=False)
                
            
                decoded_preds = processor.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)    
                
                
                predictions.append(decoded_preds[0])
                references.append(answer)
                    
                input_list.append(prompt)
                
            else:
                previous_sub_questions = []
                previous_sub_answers = []
                candidate_num = args.num_candidate
                
                qs = (bounding_boxs_context 
                        + '\n' 
                        + f"Target Entity: {bounding_boxes[instruction['target_idx']]}"
                        + "\n" 
                        + f"Generate a sub-question to classify ambiguous entities. Sub-Question:")
                
                if q2q_model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates['v1'].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, q2q_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(q2q_model.device)
                
                image_tensor = process_images([image], q2q_image_processor, model.config)[0]

                with torch.inference_mode():
                    set_seed(42)
                    
                    output_ids = q2q_model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().to(q2q_model.device),
                        image_sizes=[image.size],
                        # do_sample=True,
                        # temperature=0.9,
                        # top_p=50,# num_beams=1,
                        do_sample=True if args.num_candidate > 1 and args.num_beams < 2 else False,
                        temperature=0.9 if args.num_candidate > 1 else 1.0,
                        top_p=0.95 if args.num_candidate > 1 else None,
                        num_return_sequences=args.num_candidate if args.num_candidate > 1 else 1,
                        num_beams=args.num_beams,
                        no_repeat_ngram_size=3,
                        max_new_tokens=128,
                        use_cache=True)

                    outputs = q2q_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                    result= []
                    for i in outputs:
                        if i not in result:  
                            result.append(i)
                    
                    for sub_question in result:
                        
                        if args.checker_model_path:
                    
                            chekcer_qs = (
                                bounding_boxs_context
                                + f"\n Target Entity: {bounding_boxes[instruction['target_idx']]}"
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

                            image_tensor = process_images([image], checker_image_processor, checker_model.config)[0]

                            with torch.inference_mode():
                            
                                checker_outputs = checker_model.generate(
                                    input_ids,
                                    images=image_tensor.unsqueeze(0).half().to(checker_model.device),
                                    image_sizes=[image.size],
                                    no_repeat_ngram_size=3,
                                    max_new_tokens=128,
                                    use_cache=True,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                    )

                                checker_text = checker_tokenizer.batch_decode(checker_outputs.sequences, skip_special_tokens=True)[0].strip()
                                transition_scores = checker_model.compute_transition_scores(checker_outputs.sequences, checker_outputs.scores, normalize_logits=True)
                                confidence = np.exp(transition_scores[0][0].cpu().numpy()) # first token probablity
                
                                if 'yes' in checker_text.lower() and confidence > 0.9:
                                    pass
                                else:
                                    continue    
                                
                        answer_conversation = [
                            {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": (bounding_boxs_context 
                                                        + '\n' 
                                                        + f"Target Entity: {bounding_boxes[instruction['target_idx']]}"
                                                        + "\n" + sub_question \
                                                        + " Answer the question using Yes or No.")},
                                
                                ],
                            }
                        ]
                        
                        answer_text = answer_processor.apply_chat_template(answer_conversation, add_generation_prompt=True)
                        
                        answer_inputs = answer_processor(text=answer_text, images=image, return_tensors='pt').to(answer_model.device)
                        answer_outputs = answer_model.generate(**answer_inputs, max_new_tokens=16)
                        
                        answer_decoded_preds = answer_processor.tokenizer.batch_decode(answer_outputs[:, answer_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        answer_text = answer_decoded_preds[0].strip()
                        
                        
                        if 'yes' in answer_text.lower():
                            previous_sub_questions.append(sub_question)
                            previous_sub_answers.append('Yes')
                        
                        else:
                            previous_sub_questions.append(sub_question)
                            previous_sub_answers.append('No')
                            
                        if len(previous_sub_answers) > args.num_candidate - 1:
                            break
        
                                        
                if len(previous_sub_questions) < 1:
                    conversation = [
                        {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question + " Answer the question using a single word or phrase."
                                },
                            ],
                        }
                    ]
                    
                else:
                    conversation = []
                    for idx, (sub_q, sub_ans) in enumerate(zip(previous_sub_questions, previous_sub_answers)):
                        if idx < 1:
                            conversation.extend([
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image"},
                                        {"type": "text", "text": sub_q},
                                    ]
                                },
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": sub_ans},
                                    ]
                                },
                            ])
                        else:
                            conversation.extend([
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": sub_q},
                                    ]
                                },
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": sub_ans},
                                    ]
                                },
                            ])
                        
                    
                    conversation.extend(
                        [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question + " Answer the question using a single word or phrase."},
                            ]
                        }]
                    )
                    
        
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=prompt, images=image,return_tensors='pt').to(model.device)
                
                outputs = model.generate(**inputs,  max_new_tokens=args.max_target_length, 
                                            num_beams=args.num_beams, length_penalty=args.length_penalty, do_sample=False)
                
            
                decoded_preds = processor.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)    
            
                
                
                predictions.append(decoded_preds[0])
                references.append(answer)
                    
                input_list.append(prompt)
            
        # print(prompt)
        # print(f'predicts: {decoded_preds[0]} gold: {answer}')
            
    metrics_rusult = compute_metrics((predictions, references))

    with open(os.path.join(predictions_dir_path, "metrics_result.txt"), 'w') as f:
        json.dump(metrics_rusult, f)
    
    with open(os.path.join(predictions_dir_path, "predictions.txt"), 'w') as f:
        
        lines = [prediction+'\n' for prediction in predictions]
        
        f.writelines(lines)
        
    with open(os.path.join(predictions_dir_path, "labels.txt"), 'w') as f:
        
        lines = [label+'\n' for label in references]
        
        f.writelines(lines)
        
    with open(os.path.join(predictions_dir_path, "input_list.txt"), 'w') as f:
        
        json.dump([{'input': example } for example in input_list], f)
    
    del model
    


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    
    parser.add_argument(
        "--ref_path",
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
        "--base_model_name",
        type=str,
        default="google/t5-v1_1-large",
    )
    
    parser.add_argument(
        "--q2q_model_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        '--subquestion_file_path',
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--answer_model_path",
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
        "--prediction_dir_name",
        type=str,
        default="",
    )

    parser.add_argument(
        "--num_candidate",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="human",
    )

    
    args = parser.parse_args()
    
    inference(args)
    

if __name__ == "__main__":
    main()