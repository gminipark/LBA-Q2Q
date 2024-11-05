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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('path_to_test_data')
image_dir = "path_to_image_dir"

df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

def QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    switch = 0

    keys = sceneGraphs[str(image_id)]['objects'].keys()
    objects=sceneGraphs[str(image_id)]['objects']
    i = 0

    text1 = f"""[INST] <image> Generate short answer.
{ambiguous_question}
Answer: 
[/INST]"""
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=[image], text=text1, return_tensors="pt").to(device)
        image.save('img.jpg')
        generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        start=generated_text.find("[/INST]")
        generated_text = generated_text[start+8:]
        return generated_text, text1, ambiguous_question_answer

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""


import time
import datetime
total_start_time = time.time()

correct_li=[]
wrong_li = []

print("테스트 시작")
score = 0
output_file = 'path_to_output_file.txt
with open(output_file,'w', encoding='utf-8') as f :
    f.write("테스트 시작")
    for i in range(300):
        image_id = df[i]['image_id']
        ambiguous_question = df[i]['ambiguous_question']
        ambiguous_entity = df[i]['ambiguous_entity']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = df[i]['ambiguous_question_answer']
        no1= df[i]['no1']

        if os.path.exists(image_path):
            output, prompt, answer = QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
            output = output.lower()[:-1]
            if output == answer :
                print(f"correct{i}")
                score += 1
                correct_li.append(i+1)
            else :
                print("wrong")
                wrong_li.append(i+1)
            
            print()
            print(f"{i+1}th")
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
            f.write(f"\n점수 : {score}")
            f.write('\n')
        else:
            print(f"이미지 {image_path}이(가) 존재하지 않습니다.")
    total_elapsed_time = time.time() - total_start_time
    tot=str(datetime.timedelta(seconds=total_elapsed_time)).split(".")[0]
    f.write("\n========================================================")
    f.write(f"\n총 처리 시간 : {tot}, 최종 점수 : {score}")
    f.write(f"\nC:{correct_li}")
    f.write(f"\nW: {wrong_li}")