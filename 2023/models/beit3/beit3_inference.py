# %%
import os
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model

from beit3 import utils, modeling_finetune
from beit3_datasets import VQADataset

from evaluate import load
from collections import OrderedDict

# %%
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameter Setting
CFG = {
    'IMG_SIZE' : 480,
    'BATCH_SIZE': 32,
}

train_img_path = 'ambiguous_images'

tokenizer = XLMRobertaTokenizer(os.path.join("./", 'beit3_models', 'beit3.spm'))
# 'test_242_same_true_human.csv'
test_df = pd.read_csv(os.path.join("./", "test_367_same_true_v2.csv"), dtype={'image_id':"str"})
test_dataset = VQADataset(test_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=8)


model_config = 'beit3_large_patch16_480_vqav2'
model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=0.1,
    vocab_size=64010
)


tmp_weights = torch.load(os.path.join("./", 'beit3_models', '4_10_2e-05_betas_0.98_large_model.pt'))

finetuned_weights = OrderedDict()
for k, v in tmp_weights.items():
    finetuned_weights[k[10:]] = v   

model.load_state_dict(finetuned_weights)
model.eval()
model.to(device)


acc_metric = load("accuracy")
f1_metric = load("f1")
precision_metric = load("recall")
recall_metric = load("precision")


with torch.no_grad():
    
    predictions= []
    references= []
    for data in tqdm(test_loader, total=len(test_loader)): 
        
        images = data['image'].to(device)
        input_ids = data['input_ids'].to(device)
        padding_mask = data['padding_mask'].to(device)
        answer = data['labels']
        
        outputs = model(images, input_ids, padding_mask)
        
        prediction = torch.argmax(outputs.detach().cpu(), dim=1)

        predictions.extend(prediction)
        references.extend(answer)
    
    print(acc_metric.compute(predictions=predictions, references=references))
    print(f1_metric.compute(predictions=predictions, references=references))
    print(recall_metric.compute(predictions=predictions, references=references))
    print(precision_metric.compute(predictions=predictions, references=references))
    
    
    test_df['predctions'] = ["O" if pred == 0 else "X" for pred in predictions]
    test_df.to_csv("beit3_predictions.csv")
    

# %%



