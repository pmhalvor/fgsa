from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
import torch
import json
import logging

from dataset import Norec 

logging.basicConfig(
    filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
logging.info('----------------------------------  new run: model.py ----------------------------------')


def pad(batch, IGNORE_ID, both=False):
    longest_sentence = max([X.size(0) for X, y, z in batch])
    new_X, new_y, new_z = [], [], []

    for X, y, z in batch:
        new_X.append(torch.nn.functional.pad(
            X,
            (0, longest_sentence - X.size(0)),
            value=0)  # find padding index in bert
        )
        new_y.append(torch.nn.functional.pad(
            y,
            (0, longest_sentence - y.size(0)) if not both else(0, longest_sentence*2 - y.size(0)),
            value=IGNORE_ID)
        )
        new_z.append(torch.nn.functional.pad(
            z,
            (0, longest_sentence - z.size(0)),
            value=0)
        )

    new_X = torch.stack(new_X).long()
    new_y = torch.stack(new_y).long()
    new_z = torch.stack(new_z).long()

    return new_X, new_y, new_z



data_dir = "$HOME/data/"  # TODO hide personal info

train_dataset = Norec(
    data_path=data_dir + "norec_fine/train/"
)

dev_dataset = Norec(
    data_path=data_dir + "norec_fine/dev/"
)

# TODO split train data into train/test.


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=True,
    collate_fn=lambda batch: pad(batch=batch, IGNORE_ID=train_dataset.IGNORE_ID)
)

dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 32,
    shuffle=True,
    collate_fn=lambda batch: pad(batch=batch, IGNORE_ID=dev_dataset.IGNORE_ID)
)










#################### GRAVEYARD ####################
# with open(data_path + "norbert/bert_config.json") as f:
#     config_data = json.load(f)

# try:
#     config = BertConfig(**config_data)
# except:
#     config = BertConfig()


# model = BertModel(config)

# print(model.__dict__)
