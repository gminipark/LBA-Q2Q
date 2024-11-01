import json
import os

import torch
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import Dataset

from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image

from beit3.randaug import RandomAugment

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def build_transform(is_train, img_size):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(img_size, scale=(0.5, 1.0), interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2, 7, isPIL=True,
                augs=[
                    'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
                ]
            )
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)
        ]

    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
    t = transforms.Compose(t)

    return t


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, img_path, *, img_size=480, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = build_transform(is_train, img_size)
        self.img_path = img_path
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        img_name = os.path.join("./", self.img_path, item['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        text = "Given an ambiguous quesiton, an ambiguous entity and an intermediate question, your task is to classify whether the intermediate question is effective to clarify the ambiguous entity in the ambiguous quesiton. An ambiguous entity means that it appears multiple times in the image and cannot be distinctly identified. A good intermediate question is one that clarify a specific entity among same entities. Additionaly, A bad intermediate question is one that can't determine a target entity among the entities through the intermediate quesiton. If you think the given intermediate question is good, indicate it by answering \"Yes\". Otherwise, answer \"No\".There are only two types of answers possible: \"Yes\" and \"No\"."
        text = text + " Ambiguous question: " + item["ambiguous_question"] + "Ambiguous entity: " + item["ambiguous_entity"] + " Intermediate question: " + item["intermediate_question"] + " Short answer:"
        
        
        # text = "Ambiguous question: " + item["ambiguous_question"] +" Ambigous entity: " + item["ambiguous_entity"] + " Intermediate question: " + item["intermediate_question"] # + " Intermediate answer: " + item["intermediate_answer"]
        # text = text + " Is the intermediate question effective to clarify the ambiguous entity in the ambiguous question? Classify yes or no. Short answer: "
        
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        
            
        if 'effectiveness' in item.keys():
            label = "Yes" if item['effectiveness'] == "O" else 'No' # torch.tensor(1) if item['effectiveness'] == "O" else torch.tensor(0)
        elif 'labels' in item.keys():
            label =  torch.tensor([0]) if item['labels'] == "O" else torch.tensor([1]) # item['labels']
        else:
            label = None

        if self.is_train:
            assert label is not None 
            #answer = item['answer']
            
            # try:
            #     label = self.ans2label[answer]
            #     one_hots = torch.nn.functional.one_hot(label, num_classes=3129)
            # except KeyError:    # 3129개 이외의 클래스에 해당하는 답변 예외 처리
            #     one_hots = torch.tensor([0]*3129)

            # labels = self.tokenizer(text_target = label, max_length=2, return_tensors='pt', padding=True)['input_ids']
            
    
        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(),
            'padding_mask': inputs['attention_mask'].squeeze().logical_not().to(int),
            'labels' : label.squeeze() if label is not None else None
        }