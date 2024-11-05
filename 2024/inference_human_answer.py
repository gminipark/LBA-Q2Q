import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForPreTraining, LlavaNextProcessor, AutoTokenizer
from PIL import Image, ImageDraw
import os
import transformers
import json
import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('path_to_test_set')
image_dir = "path_to_image_dir"

df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

def CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
   
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    image = Image.open(image_path)

    i = 0
    q_count = 0
    context = ""
    while q_count < 3 :
        switch = 0  
        
        additional_question2 = f"""[INST] <image>
You receive an original question that refers to an ambiguous entity, making it difficult to identify which one is being discussed. Your task is to create a new simple yes/no question, using the provided context, that helps clarify which {ambiguous_entity} is being referred to in the original question.

In your new question, make sure to clearly differentiate the target {ambiguous_entity} from the other non-target {ambiguous_entity} instances in the image. Use distinguishing features such as the {ambiguous_entity}'s attributes, appearance, or its relative position to other non-target {ambiguous_entity} instances to help identify the correct one.

Once you've generated the question, answer it without explicitly stating the answer. If the answer is 'yes', present the question you created. If the answer is 'no', generate a different question and try again.

Ensure that the yes/no question you create can reliably distinguish the target {ambiguous_entity} from the others. Avoid repeating any question that the ASSISTANT has already asked.

The original question: ' {ambiguous_question}'
The ambiguous entity: '{ambiguous_entity}'

{context}
ASSISTANT: [/INST]"""
        generated_question = generate_output(image, ambiguous_question, ambiguous_entity, additional_question2, entity_id, switch, tar_coordinate)
        context += f'ASSISTANT: {generated_question[0]}\n'

        print(generated_question[0])
        User_answer = input("USER:")
        case_switch = User_answer
        case_switch = case_switch.lower()


        if 'yes' in case_switch :
            context += 'USER: Yes\n'
            break
    
        else : 
            context += 'USER: No.\n'
        q_count += 1 
    additional_question_final = f"""[INST] <image>
{context}
USER: {ambiguous_question} Answer in short answer.
ASSISTANT: [/INST]"""
    final_output = generate_output(image, ambiguous_question, ambiguous_entity, additional_question_final, entity_id, switch, tar_coordinate)
    return final_output[0], final_output[1], ambiguous_question_answer, context



def generate_output(image, ambiguous_question, ambiguous_entity, additional_question, entity_id, switch, tar_coordinate):

    inputs = processor(images=[image], text=additional_question, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    start = generated_text.find("[/INST]")
    generated_text = generated_text[start+8:]

    return generated_text, additional_question


import time
import datetime
total_start_time = time.time()
output_file = 'path_to_output.txt'
correct_li=[]
wrong_li=[]
print("테스트 시작")
score = 0
with open(output_file,'w', encoding='utf-8') as f :
    f.write("테스트 시작")
    for i in range(300):
        image_id = df[i]['image_id']
        ambiguous_question = df[i]['ambiguous_question']
        ambiguous_entity = df[i]['ambiguous_entity']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = df[i]['ambiguous_question_answer']
        target_question = df[i]['additional_question']
        no1= df[i]['no1']

        if os.path.exists(image_path):
            output, prompt, answer, contexts = CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
            output = output.lower()[:-1] 
            if answer == output :
                f.write("\ncorrect")
                score += 1
                correct_li.append(i+1)
            else :
                f.write("\nwrong")
                wrong_li.append(i+1)
            print()
            print(f"{i+1}th")
            print(contexts[:-1])
            print(ambiguous_question)
            print("output",output)
            print("answer",answer)
            print("score",score)
            f.write(f"\n{i+1}th")
            f.write(f"\n이미지 ID: {image_id}")
            f.write(f"\n엔티티 ID: {entity_id}")
            f.write(f"\n입력 질문:  {ambiguous_question}")
            f.write(f"\n출력 답: {output}")
            f.write(f"\n실제 답: {answer}")
            f.write(f"\n정답 질문: {target_question}")
            f.write(f"\n{contexts}")
            f.write(f"\n점수 : {score}")
            f.write('\n')
        else:
            print(f"이미지 {image_path}이(가) 존재하지 않습니다.")
            f.write(f"\n이미지 {image_path}이(가) 존재하지 않습니다.")

    total_elapsed_time = time.time() - total_start_time
    tot=str(datetime.timedelta(seconds=total_elapsed_time)).split(".")[0]
    f.write("\n========================================================")
    f.write(f"\n총 처리 시간 : {tot}, 최종 점수 : {score}")
    f.write(f"\nC : {correct_li}")
    f.write(f"\nW : {wrong_li}")