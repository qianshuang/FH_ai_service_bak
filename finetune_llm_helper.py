# -*- coding: utf-8 -*

import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载分类模型
finetuned_model_path = "/mnt/models/finetune/chinese-macbert-base/"
with open(finetuned_model_path + 'label_to_id.json', 'r') as f:
    label_to_id = json.load(f)
id_to_label = {v: k for k, v in label_to_id.items()}

finetunedM = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path, device_map={"": "cuda:1"})
tokenizerM = AutoTokenizer.from_pretrained(finetuned_model_path)
print("TC model loaded successfully...")


def get_tc_res(text):
    tokens = tokenizerM([text], padding="max_length", truncation=True, return_tensors="pt").to("cuda:1")
    outputs = finetunedM(**tokens)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores, predicted_labels = torch.max(predictions, dim=-1)
    return [id_to_label[i] for i in predicted_labels.tolist()][0], scores.tolist()[0]
