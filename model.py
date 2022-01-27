from transformers import BertConfig, BertModel
import torch 
import json

data_path = "$HOME/data/"  # TODO hide personal info

with open(data_path + "norbert/bert_config.json") as f:
    config_data = json.load(f)

try:
    config = BertConfig(**config_data)
except:
    config = BertConfig()


model = BertModel(config)

print(model.__dict__)
