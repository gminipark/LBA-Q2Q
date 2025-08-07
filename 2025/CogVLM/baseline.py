import pandas as pd
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
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
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

CLIENT_ID = "CLIENT_ID"
my_api_key= "api key"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

csv_path = "test data.csv"
image_dir = "image dir path"

df = pd.read_csv(csv_path)

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
device_map = infer_auto_device_map(model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
model = load_checkpoint_and_dispatch(
    model,
    '/root/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c',
    device_map=device_map,
)
model = model.eval()

fasttext.util.download_model('en', if_exists='ignore') 
ft_model = fasttext.load_model('cc.en.300.bin')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') 


def QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    keys = sceneGraphs[str(image_id)]['objects'].keys()
    objects=sceneGraphs[str(image_id)]['objects']

    text1 = f"""USER: {ambiguous_question} Answer using only one or two words. No full sentences. No explanations. No extra words.
ASSISTANT:"""
    prompt = text1
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = model.build_conversation_input_ids(tokenizer, query=text1, history=[], images=[image])  
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(outputs[0])
        return generated_text, text1, ambiguous_question_answer

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""

def cossim(ft_model, w1, w2):
    model = ft_model
    w1_words = w1.split()
    w2_words = w2.split()

    if not w1_words or not w2_words:
        return 0.0 

    vec1 = np.mean([model.get_word_vector(word) for word in w1_words], axis=0)
    vec2 = np.mean([model.get_word_vector(word) for word in w2_words], axis=0)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


import time
import datetime
total_start_time = time.time()
output_file = 'output path.txt'
correct_li=[]
wrong_li=[]
cos_score_li = []
max_items = len(df)
print("테스트 시작")
score = 0
sim_score = 0
import shutil
with open(output_file,'w', encoding='utf-8') as f :
    f.write(f"prompt: {prompt}\n")
    f.write("테스트 시작")
    for i in range(max_items):
        row = df.iloc[i]

        image_id = row['image_id']
        ambiguous_question = row['ambiguous_question']
        ambiguous_entity = row['ambiguous_entity']
        entity_id = row['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = row['ambiguous_question_answer']
        target_question = row['additional_question']

        if os.path.exists(image_path):
            output, prompt, answer = QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
            output = output.lower()[:-4] 
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
            print(ambiguous_question)
            print("output",output)
            print("answer",answer)
            print("score",score)
            print("cossim score 값", cossim_score)
            print("cossim score 점수", sim_score)
            f.write(f"\n{i+1}th")
            f.write(f"\n이미지 ID: {image_id}")
            f.write(f"\n엔티티 ID: {entity_id}")
            f.write(f"\n입력 질문:  {ambiguous_question}")
            f.write(f"\n출력 답: {output}")
            f.write(f"\n실제 답: {answer}")
            f.write(f"\n정답 질문: {target_question}")
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