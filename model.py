from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
from transformer import Transformer  # TODO change to BertSimple
import torch
import json
import logging
import sys

from dataset import Norec, NorecOneHot 


####################  config  ####################
logging.basicConfig(
    filename='out.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))  # didnt work as expected
logging.info('----------------------------------  new run: model.py ----------------------------------')

DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/"  # TODO hide personal info
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
###################################################

#################### refator out ####################
def pad(batch):  # removed: both=False
    """
    Pad batches according to largest sentence.

    A sentence in the batch has shape [3, sentence_length] and looks like:
        (
            tensor([  102,  3707, 29922,  1773,  4658, 13502,  1773,  3325,  3357, 19275,
                    3896,  3638,  3169, 10566,  8030, 30337,  2857,  3707,  4297, 24718,
                    9426, 29916, 28004,  8004, 30337, 15271,  4895, 10219,  6083,  4297,
                    26375, 20322, 26273,   103]), 
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
            tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        )
    """
    longest_sentence = max([ids.size(0) for ids, _, _ in batch])
    padded_ids, padded_masks, padded_labels = [], [], []

    for id, mask, label in batch:
        padded_ids.append(
            torch.nn.functional.pad(
                id,
                (0, longest_sentence - id.size(0)),
                value=0  # padding token can vary between Berts
            )
        )
        padded_masks.append(
            torch.nn.functional.pad(
                mask,
                (0, longest_sentence - mask.size(0)),  # removed:  if not both else(0, longest_sentence*2 - y.size(0))
                value=0  # no longer last index in 
            )
        )
        padded_labels.append(
            torch.nn.functional.pad(
                label,
                (0, longest_sentence - label.size(0)),
                value=-1  # NOTE cannot pad with 0 since thats used as label O FIXME make sure negative works
            )
        )

    ids = torch.stack(padded_ids).long()
    masks = torch.stack(padded_masks).long()
    labels = torch.stack(padded_labels).long()

    return ids, masks, labels
#####################################################



# load train/dev/test data so every build has complete result set
train_dataset = NorecOneHot(data_path=DATA_DIR + "norec_fine/train/", proportion=0.3)
test_dataset = NorecOneHot(data_path=DATA_DIR + "norec_fine/test/")
dev_dataset = NorecOneHot(data_path=DATA_DIR + "norec_fine/dev/")

# for i in range(3):
#     print('------- Index {} -------'.format(i))
#     print(train_dataset.sentence[i])
#     print(train_dataset.labels[i])
#     print('Tensorfied: ')
#     print(train_dataset[i])
#     print('------------------------\n')


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=True,
    collate_fn=lambda batch: pad(batch)
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 1,  # for predict to work
    shuffle=True,
    collate_fn=lambda batch: pad(batch)
)

dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 1,  # for predict to work
    shuffle=True,
    collate_fn=lambda batch: pad(batch)
)


# for i, batch in enumerate(train_loader):
#     print('------- Index {} -------'.format(i))
#     print(batch[0].shape)
#     print(batch[1].shape)
#     print(batch[2].shape)
#     print('------------------------\n')

#     if i>3:
#         quit()

model = Transformer(
    NORBERT='ltgoslo/norbert',
    tokenizer=train_dataset.tokenizer,
    num_labels=9, 
    IGNORE_ID=-1, 
    device=DEVICE,
    epochs=10,
    lr_scheduler=False,
    factor=0.1,
    lrs_patience=2,
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr=0.0001,
    momentum=0.9,
    epoch_patience=1,
    label_indexer=None,
    optmizer='AdamW'
)

print('Fitting model...')
model.fit(train_loader=train_loader, verbose=True, dev_loader=dev_loader)

print('Evaluating model...')
binary_f1, propor_f1 = model.evaluate(test_loader)
print("Binary F1: {}".format(binary_f1))
print("Proportional F1: {}".format(propor_f1))




#################### GRAVEYARD ####################
# with open(data_path + "norbert/bert_config.json") as f:
#     config_data = json.load(f)

# try:
#     config = BertConfig(**config_data)
# except:
#     config = BertConfig()


# model = BertModel(config)

# print(model.__dict__)
