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
from openai import OpenAI
import base64
import requests
import time
import fasttext.util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('lba_test')
image_dir = "ambiguous_images"


df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

fasttext.util.download_model('en', if_exists='ignore') 
ft_model = fasttext.load_model('cc.en.300.bin') 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') 

def Boxing(image_id):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    image = Image.open(image_path)
    
    locations = sceneGraphs[str(image_id)]['objects'][str(entity_id)]
    width, height = sceneGraphs[str(image_id)]['width'], sceneGraphs[str(image_id)]['height']
    x, y, w, h = locations['x'], locations['y'], locations['w'], locations['h']
    tar_coordinate = (x, y, w, h)

    image_box = image.copy()
    draw = ImageDraw.Draw(image_box)
    draw.rectangle([x, y, x+w, y+h], outline="red",width=3)
    image_box.save(f'boxed_image.jpg')


def CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }   
    Boxing(image_id)
    boxed_img_path = 'boxed_image.jpg'
    base64_image = encode_image(boxed_img_path)

    i = 0
    q_count = 0
    context = ""
    switch = 0
    output, _, answer, confi = QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
    if confi < 0.9 :
        pass
    else : 
        return confi, output, "tmp", answer, "tmp"
    while q_count < 3 :
        switch = 0  
        # 1번 질문
        QG_prompt = f"""[INST] <image>
You receive an original question that has an ambiguous entity difficult to specify.
The task is to use the provided context, and generate a new yes/no question asking about the {ambiguous_entity} referred to the original question.
In the generated question, you must distinguish the {ambiguous_entity} from the other '{ambiguous_entity}' in the image.
By answering the new yes/no question, one can identify which '{ambiguous_entity}' is referred to the original question.
When generating a new question, you can use the attributes of the {ambiguous_entity} or relative location to the other {ambiguous_entity} in the image.
If you created a question, answer it without saying it directly. If the answer is 'yes', show the question you created, and if it is 'no', do it again.
Make sure that the yes/no question allows you to distinguish {ambiguous_entity} from the other {ambiguous_entity}.

The original question: '{ambiguous_question}'
The ambiguous entity: '{ambiguous_entity}'

{context}
ASSISTANT: [/INST]"""
        prompt = QG_prompt
        generated_question = generate_output(image_id, ambiguous_question, ambiguous_entity, QG_prompt, entity_id, switch)
        context += f'USER: {generated_question[0]}\n'


        switch = 'check'

        sys_prompt  = f"""You are an AI trained to support visual recognition tasks. Your job is to look at the image with s red bounding box around specific object and answer the question about it. You have to analyze the object inside the red bbox and user tells you the name of the object in the input text. Based on this object, you answer the question. 
Your goal
You can focus only on the object within the red bbox. Focus only on objects and properties when answering a question, without mentioning the bounding boxes in the image. 
Don't generate any new question.

When asked about direction or relative position, think twice and answer correctly.
If the question includes 'or', answer which one of the two options is correct.
If the question is wrong, you should correct it.
If the question is asking about the other {ambiguous_entity} which is not within the red bbox, you should correct it.

In addition, the answer should be simple and concise."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {my_api_key}"
        }
        payload = {
            "model": "gpt-4o",  
            "messages": [
                {
                    "role": "system",
                    "content": sys_prompt               },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", # user prompt
                            "text": f"""Here is a question about the given image. The object in question is "{ambiguous_entity}".
Original question : {ambiguous_question}

The "{ambiguous_entity}" in the sub-question is in the red box in the image. Answer the sub-question. You can also give additional hints, if necessary, to help answer the original question in a sentence.
sub-question : {generated_question}
Answer : """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature":0.1,
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
        try:
            response_json = response.json() 
        except requests.exceptions.JSONDecodeError:
            print(f"[JSON Decode Error] 응답이 JSON 형식이 아닙니다:\n{response.text}")
            return "Error", "Error", "prompt", ambiguous_question_answer, "No context"

        if 'choices' not in response_json:
            print(f"[Response Format Error] 'choices' not found in response: {response_json}")
            return "Error", "Error", "prompt", ambiguous_question_answer, "No context"
            
        case_switch = response_json['choices'][0]['message']['content'].lower()

        # yes인 경우
        if 'yes' in case_switch :
            context += f'ASSISTANT: Yes\n'
            break
        # no인 경우
        else : 
            context += f'ASSISTANT: {case_switch}\n'
        q_count += 1 

    additional_question_final = f"""[INST] <image>
{context}

USER: {ambiguous_question} Answer in one or two words.
ASSISTANT: [/INST]"""
    final_prompt = additional_question_final
    final_output = generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question_final, entity_id, switch)

    return confi, final_output[0], final_output[1], ambiguous_question_answer, context


def generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question, entity_id, switch):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        # 이미지 로드 및 RGB로 변환
        image = Image.open(image_path)


        # 프로세서 입력값 생성 (이미지를 PIL 이미지 객체가 아닌 텐서로 변환)
        inputs = processor(images=[image], text=additional_question, return_tensors="pt").to(device)

        image.save('img.jpg')
        # 모델 출력 생성
        generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id, temperature=0, do_sample=False, top_k=1, top_p=0)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        start = generated_text.find("[/INST]")
        generated_text = generated_text[start+8:]
        return generated_text, additional_question

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""

def QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    switch = 0
    i = 0

    # 1번 질문
    text1 = f"""[INST] <image>
    USER: {ambiguous_question} Answer in a short answer.
    ASSISTANT: [/INST]"""
    prompt = text1
    image_path = os.path.join(image_dir, f"{image_id}.jpg")

    # 모델 출력 생성
    image = Image.open(image_path).convert("RGB")
    # 프로세서 입력값 생성 (이미지를 PIL 이미지 객체가 아닌 텐서로 변환)
    inputs = processor(images=[image], text=text1, return_tensors="pt").to(device)
    image.save('img.jpg')
    # 모델 출력 생성
    generated_outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,  # logits을 포함한 출력을 요청
        output_scores=True  # 토큰별 확률(score) 반환
    )

    # 생성된 텍스트 디코딩
    generated_ids = generated_outputs.sequences[0]  # 첫 번째 샘플의 생성된 토큰 ID
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    start = generated_text.find("[/INST]") + len("[/INST]")
    generated_text = generated_text[start:].strip()
    input_length = inputs["input_ids"].shape[1]  # 입력 텍스트의 길이
    generated_only_ids = generated_ids[input_length:]  # 모델이 생성한 부분만 추출

    # 생성된 모든 토큰의 logits (각 step마다 logits이 있음)
    scores = generated_outputs.scores  # 각 토큰의 logits 리스트

    # "[/INST]" 이후의 토큰에 대한 Confidence 계산
    post_inst_confidences = []
    for i, token_id in enumerate(generated_only_ids):
        softmax_probs = torch.softmax(scores[i], dim=-1).squeeze()  # 차원 축소 추가

        # 토큰 ID가 softmax_probs 범위 내에 있는지 확인
        if token_id >= softmax_probs.shape[0]:
            print(f"Warning: token_id {token_id} out of range. Skipping...")
            continue

        confidence = softmax_probs[token_id].item()  # 해당 토큰의 확률 추출
        post_inst_confidences.append(confidence)

    # 평균 Confidence 계산
    average_confidence = sum(post_inst_confidences) / len(post_inst_confidences) if post_inst_confidences else 0.0

    return generated_text, text1, ambiguous_question_answer, average_confidence


def cossim(ft_model, w1, w2):
    model = ft_model
    vec1 = np.mean([model.get_word_vector(word) for word in w1.split()], axis=0)
    vec2 = np.mean([model.get_word_vector(word) for word in w2.split()], axis=0)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

import time
import datetime
total_start_time = time.time()
output_file = 'output path.txt'
correct_li=[]
wrong_li=[]
cos_score_li=[]
print("테스트 시작")
score = 0
sim_score=0

import shutil
with open(output_file,'w', encoding='utf-8') as f :
    f.write("테스트 시작")
    for i in range(1000): # 1000개 test
        image_id = df[i]['image_id']
        ambiguous_question = df[i]['ambiguous_question']
        ambiguous_entity = df[i]['ambiguous_entity']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = df[i]['ambiguous_question_answer']
        target_question = df[i]['additional_question']
        no1= df[i]['no1']


        if os.path.exists(image_path):
            confi, output, prompt, answer, contexts = CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
            output = output.lower()
            output = output.strip()
            if answer == output :
                f.write("\ncorrect")
                score += 1
                correct_li.append(i+1)
            else :
                f.write("\nwrong")
                wrong_li.append(i+1)
            cossim_score = round(cossim(ft_model, answer, output),4)
            cos_score_li.append(cossim_score)
            if cossim_score >= 0.7 :
                sim_score += 1

            print()
            print(f"{i+1}th")
            print(contexts[:-1])
            print(ambiguous_question)
            print("output",output)
            print("answer",answer)
            print("confi",confi)
            print("score",score)
            print("cossim score 값", cossim_score)
            print("cossim score 점수", sim_score)
            f.write(f"\n{i+1}th")
            f.write(f"\n이미지 ID: {image_id}")
            f.write(f"\n엔티티 ID: {entity_id}")
            f.write(f"\n입력 질문:  {ambiguous_question}")
            f.write(f"\nconfi: {confi}")
            f.write(f"\n출력 답: {output}")
            f.write(f"\n실제 답: {answer}")
            f.write(f"\n정답 질문: {target_question}")
            f.write(f"\n{contexts}")
            f.write(f"\n점수 : {score}")
            f.write(f"\ncossim 값 : {cossim_score}")
            f.write(f"\ncossim 점수 : {sim_score}")
            f.write('\n')
        else:
            print(f"이미지 {image_path}이(가) 존재하지 않습니다.")
            f.write(f"\n이미지 {image_path}이(가) 존재하지 않습니다.")

    print("cossim 평균 :", sum(cos_score_li)/len(cos_score_li))
    f.write(f"\ncossim 평균 : {sum(cos_score_li)/len(cos_score_li)}")
    total_elapsed_time = time.time() - total_start_time
    tot=str(datetime.timedelta(seconds=total_elapsed_time)).split(".")[0]
    f.write("\n========================================================")
    f.write(f"\n총 처리 시간 : {tot}, 최종 점수 : {score}, 최종 cossim 점수 : {sim_score}")
    f.write(f"\nC:{correct_li}")
    f.write(f"\nW: {wrong_li}")