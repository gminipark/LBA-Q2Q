# %%
import os
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model

from beit3 import utils, modeling_finetune
from beit3_datasets import VQADataset

# %%
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameter Setting
CFG = {
    'IMG_SIZE': 480,
    'EPOCHS': 10,
    'LEARNING_RATE': 2e-5,
    'BATCH_SIZE': 8,
    'SEED': 42
}

# %%
# Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

# %%
# Data Load
train_df = pd.read_csv(os.path.join("./", 'train_6400_same_true_aug_v2.csv'), dtype={'image_id':"str"})
train_img_path = 'ambiguous_images'

# dataset & dataloader
tokenizer = XLMRobertaTokenizer(os.path.join("./", 'beit3_models', 'beit3.spm'))
train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=True)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=8)

test_df = pd.read_csv(os.path.join("./", 'test_367_same_true_v2.csv'), dtype={'image_id':"str"})
test_dataset = VQADataset(test_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=False)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=8)


# %%
model_config = 'beit3_large_patch16_480_vqav2'
model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=0.1,
    vocab_size=64010
)

utils.load_model_and_may_interpolate(
    ckpt_path=os.path.join("./", 'beit3_models', 'beit3_large_indomain_patch16_224.zip'),
    model=model,
    model_key='model|module',
    model_prefix=''
)

model = torch.compile(model)

# %%
criterion = nn.BCEWithLogitsLoss(reduction='mean')

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.9, 0.98), weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=len(train_loader) * int(CFG["EPOCHS"] * 0.1),
    num_training_steps=len(train_loader) * CFG["EPOCHS"]
)

def compute_acc(predictions, references):
    
    total_len = len(predictions)
    same_count = 0
    for prediction, reference in zip(predictions, references):
        if prediction == reference:
            same_count += 1
    
    return same_count / total_len

model.train()
model.to(device)
for epoch in range(1, CFG['EPOCHS']+1):
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        input_ids = data['input_ids'].to(device)
        padding_mask = data['padding_mask'].to(device)
        answer = data['labels'].to(device)
        
        optimizer.zero_grad()

        outputs = model(images, input_ids, padding_mask)
        
        
        #print(outputs.view(-1, outputs.size(-1)).size())
        #print(answer.view(-1).size())
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), answer.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch}/{CFG["EPOCHS"]}], Train Loss: [{avg_loss:.5f}]')

    model.eval()
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
        
        print("Acc: ", compute_acc(predictions, references) * 100)
    
    model.train()
        
    torch.save(
        model.state_dict(),
        os.path.join("./", 'beit3_models', f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".format(CFG["LEARNING_RATE"])}_betas_0.98_large_model.pt')
    )

# %%



