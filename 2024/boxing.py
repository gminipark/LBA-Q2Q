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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('path_to_test_set')
image_dir = "path_to_image_dir"

df = dataset

def Boxing(image_id, entity_id):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    image = Image.open(image_path)
    locations = sceneGraphs[str(image_id)]['objects'][str(entity_id)]
    width, height = sceneGraphs[str(image_id)]['width'], sceneGraphs[str(image_id)]['height']
    x, y, w, h = locations['x'], locations['y'], locations['w'], locations['h']
    x1, x2 = round(x/width,4), round((x+w)/width,4)
    y1, y2 = round(y/height,4), round((y+h)/height,4)
    tar_coordinate = (x, y, w, h)

    image_box = image.copy()
    draw = ImageDraw.Draw(image_box)
    draw.rectangle([x, y, x+w, y+h], outline="red",width=3)
    image_box.save(f'path_to_bosed_image_dir/{image_id}.jpg')
   

import shutil
with open(output_file,'w', encoding='utf-8') as f :
    f.write("테스트 시작")
    for i in range(300):
        image_id = df[i]['image_id']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        target_question = df[i]['additional_question']
        no1= df[i]['no1']


        if os.path.exists(image_path):
            Boxing(image_id, entity_id)