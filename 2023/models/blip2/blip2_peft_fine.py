# %%
# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import csv

# %%

# %%
# Let's define the LoraConfig
config = LoraConfig(
    r=128,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)

model_name_or_path = 'Salesforce/blip2-flan-t5-xl'
cache_dir = "./" + model_name_or_path.split('/')[-1]

dtype = torch.float16

# We load our model and processor using `transformers`
processor = AutoProcessor.from_pretrained(model_name_or_path,cache_dir=cache_dir)
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path,cache_dir=cache_dir, torch_dtype=dtype)

# Get our peft model and print the number of trainable parameters
model = get_peft_model(model, config)
model.print_trainable_parameters()

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = Model(model)

model.to(device)
model.train()

# %%
train_dataset = load_dataset("csv", data_files={"train" : "./train_6400_same_true_aug_v2.csv"}, split="train")
test_dataset = load_dataset("csv", data_files={"test" : "./test_367_same_true_v2.csv"}, split="test")

print(train_dataset)
print(test_dataset)

# %%
class ImageTextClassificationDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = Image.open("./images/"+str(item['image_id'])+".jpg")
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = "Ambiguous question: " + item["ambiguous_question"] +" Ambigous entity: " + item["ambiguous_entity"] + " Intermediate question: " + item["intermediate_question"] # + " Intermediate answer: " + item["intermediate_answer"]
        encoding["text"] = encoding['text'] + " Is the intermediate question effective to clarify the ambiguous entity in the ambiguous question? Classify yes or no. Short answer: "
        if 'effectiveness' in item.keys():
            encoding['label'] = "yes" if item['effectiveness'] == "O" else 'no' # torch.tensor(1) if item['effectiveness'] == "O" else torch.tensor(0)
        elif 'labels' in item.keys():
            encoding['label'] =  "Yes" if item['labels'] == "O" else 'No' # item['labels']
        else:
            encoding['label'] = encoding['text']

        
        if "t5" in self.processor.tokenizer.name_or_path:
            encoding['decoder_input_ids'] = torch.tensor([self.processor.tokenizer.pad_token_id])
        
        return encoding


def collator(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key == "text":
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
            
        elif key == "label":
            labels = processor.tokenizer([example['label'] for example in batch], padding=True, add_special_tokens=True, return_tensors='pt')
            processed_batch['label'] = labels['input_ids']
        else:
            processed_batch[key] = torch.stack([example[key] for example in batch])
     
    return processed_batch

# %%

train_dataset = ImageTextClassificationDataset(train_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collator)

test_dataset = ImageTextClassificationDataset(test_dataset, processor)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, collate_fn=collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)



# %%
import numpy as np
from torch import nn
epoch_loss_list = []

#criterion = nn.CrossEntropyLoss(reduction='mean')

# with open("./ambiguous_questions_test.csv", 'r') as f:
#     reader = csv.reader(f)
#     lines = [line for line in reader]

def compute_acc(predictions, references):
    
    total_len = len(predictions)
    same_count = 0
    for prediction, reference in zip(predictions, references):
        if prediction == reference:
            same_count += 1
    
    return same_count / total_len

for epoch in range(10):
    print("Epoch:", epoch)
    epoch_loss = []
    for idx, batch in enumerate(tqdm(train_dataloader)):
        
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, dtype=dtype)
        labels = batch.pop("label").to(device)
        if "t5" in model_name_or_path:
            decoder_input_ids = batch.pop("decoder_input_ids").to(device)
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        
        else:
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        
        #print(labels)
        #print(outputs)
        
        # loss = criterion(outputs, labels)
        loss = outputs.loss
        #print(loss.item())
        #loss = torch.mean(outputs)
        
        epoch_loss.append(loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        #if idx % 10 == 0:
        #    generated_output = model.generate(pixel_values=pixel_values, input_ids=input_ids)
        #    print(processor.batch_decode(generated_output, skip_special_tokens=True))
    
    print(np.mean(epoch_loss))
    
    
    model.eval()
    with torch.no_grad():
        epoch_outputs = []
        gold_references = []
        # metric = load("accuracy")
        for idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, dtype=dtype)
            labels = batch.pop("label").to(device)
            # if "t5" in model_name_or_path:
            #     decoder_input_ids = batch.pop("decoder_input_ids").to(device)
            #     logits = model(pixel_values, input_ids, decoder_input_ids)
            # else:
            outputs = model.generate(pixel_values=pixel_values, input_ids=input_ids)
            predictions = processor.batch_decode(outputs, skip_special_tokens=True)
            references = processor.batch_decode(labels, skip_special_tokens=True)
            # metric.add_batch(predictions=predictions, references=references)
            
            epoch_outputs += predictions #processor.batch_decode(generated_output, skip_special_tokens=True)
            gold_references += references
            
        #accuracy = metric.compute()
        print(epoch_outputs)
        print(gold_references)
        print(compute_acc(epoch_outputs , gold_references))
        
    # with open ("./test_{}.csv".format(epoch), 'w') as f:
        
    #     writer = csv.writer(f)
    #     for idx, line in enumerate(lines):
    #         if idx == 0:
    #             writer.writerow(line)
    #         else:
    #             line.append(epoch_outputs[idx-1])
    #             writer.writerow(line)
                
    model.train()            
                

# %% [markdown]
# 

# %%



