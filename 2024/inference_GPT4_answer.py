import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForPreTraining, LlavaNextProcessor, AutoTokenizer
from PIL import Image
import os
import transformers
import json
import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
import base64
import requests
import time

CLIENT_ID = "client_id"
my_api_key="my_api_key"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('path_to_test_set')
image_dir = "path_to_image_dir"

df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

def create_image_url_dict(complete_images_file, image_urls_file):
    image_url_dict = {}

    with open(complete_images_file, 'r') as img_file, open(image_urls_file, 'r') as url_file:
        image_lines = img_file.readlines() 
        url_lines = url_file.readlines()          
        
        for img_line, url_line in zip(image_lines, url_lines):
            img_name = img_line.strip() 
            img_url = url_line.strip()
            image_url_dict[img_name] = img_url  

    return image_url_dict

complete_images_file = 'path_to_image_num.txt'
image_urls_file = 'path_to_image_urls.txt'

image_url_dict = create_image_url_dict(complete_images_file, image_urls_file)

print(image_url_dict)


def CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }   
    key = f"{image_id}.jpg"
    image_url = image_url_dict[key]

    locations = sceneGraphs[str(image_id)]['objects'][str(entity_id)]
    width, height = sceneGraphs[str(image_id)]['width'], sceneGraphs[str(image_id)]['height']
    x, y, w, h = locations['x'], locations['y'], locations['w'], locations['h']
    x1, x2 = round(x/width,4), round((x+w)/width,4)
    y1, y2 = round(y/height,4), round((y+h)/height,4)
    tar_coordinate = (x, y, w, h)

    q_count = 0
    context = ""
    while q_count < 3 :
        switch = 0  
        additional_question2 = f"""[INST] <image>
You are presented with an original question containing an ambiguous entity that is difficult to distinguish.
Your task is to generate a clear yes/no question that helps specify the ambiguous entity referred to in the original question.

1. Use the context from the previous conversation to form a new yes/no question that distinguishes the ambiguous entity from other instances of '{ambiguous_entity}'.
2. In your question, leverage attributes of the {ambiguous_entity} or its relation to other entities mentioned in the conversation.
3. If the yes/no question you generate allows successful identification of the ambiguous entity, display the question.
4. If not, repeat the process by creating another question.
Make sure each yes/no question uniquely identifies the {ambiguous_entity} referenced in the original question.

The original question: '{ambiguous_question}'
The ambiguous entity: '{ambiguous_entity}'

Context: {context}
ASSISTANT: [/INST]"""
        generated_question = generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question2, entity_id, switch, tar_coordinate)
        context += f'ASSISTANT: {generated_question[0]}\n'


        #GPT 4o
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {my_api_key}"
        }
        payload = {
            "model": "gpt-4o",  
            "messages": [
                {
                    "role": "system",
                    "content": """You are an AI trained to assist with visual recognition tasks. Your job is to look at images provided with red bounding boxes (bbox) around specific objects. You will analyze the object inside the red bbox, which will be named in the input. Based on this object, answer the user’s question.

You are only allowed to focus on the object within the red bbox. The user will provide additional context or questions related to the object. Do not mention any bounding boxes in the image, just focus on the object and its properties when answering the question. 
Additionally, your answers should be brief and concise, responding in a simple and direct manner.
"""                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Here is an image with an object. The object in question is "{ambiguous_entity}".

Question: {ambiguous_question}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url":image_url}
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature":0,
            "top_p":1
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
        case_switch = response.json()['choices'][0]['message']['content'].lower()

        if 'yes' in case_switch :
            context += 'USER: Yes\n'
            break
        else : 
            context += f'USER: {case_switch}\n'
        q_count += 1 

    additional_question_final = f"""[INST] <image>
{context}
USER: {ambiguous_question} Answer in short answer.
ASSISTANT: [/INST]"""

    final_output = generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question_final, entity_id, switch, tar_coordinate)
    return final_output[0], final_output[1], ambiguous_question_answer, context


def generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question, entity_id, switch, tar_coordinate):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        image = Image.open(image_path)

        if switch == 'check' :
            image = np.array(image)
            x, y, w, h = tar_coordinate
            roi = image[y:y+h, x:x+w]
            black = 255*np.ones(image.shape, dtype = np.uint8)
            black[y:y+h, x:x+w] = roi
            image = Image.fromarray(black)

        inputs = processor(images=[image], text=additional_question, return_tensors="pt").to(device)

        image.save('img.jpg')
        generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        start = generated_text.find("[/INST]")
        generated_text = generated_text[start+8:]
        return generated_text, additional_question

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""

import time
import datetime
total_start_time = time.time()
output_file = 'path_to_output.txt'
correct_li=[]
wrong_li=[]
print("테스트 시작")
score = 0
import shutil
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
    f.write(f"\nC:{correct_li}")
    f.write(f"\nW: {wrong_li}")