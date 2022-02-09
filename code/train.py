from torch.utils.data import DataLoader
import logging
import torch

## LOCAL
from dataset import Norec, NorecOneHot 
from model import Transformer 
from model import BertSimple
from utils import pad
import config 


####################  config  ####################
config.log_train(name='BertSimple-dev')
DATA_DIR = config.DATA_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
###################################################


# load train/dev/test data so every build has complete result set
logging.info("Loading datasets..")
train_dataset = NorecOneHot(
    data_path=DATA_DIR + "train/", 
    ignore_id=-1,
    proportion=0.05,
    )
test_dataset = NorecOneHot(
    data_path=DATA_DIR + "test/", 
    ignore_id=-1,
    proportion=0.05,
    tokenizer=train_dataset.tokenizer,
    )
dev_dataset = NorecOneHot(
    data_path=DATA_DIR + "dev/", 
    ignore_id=-1,
    proportion=0.05,
    tokenizer=train_dataset.tokenizer,
    )


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=True,
    collate_fn=lambda batch: pad(batch)
)
# test_loader = DataLoader(
#     dataset = test_dataset,
#     batch_size = 32,  # for predict to work
#     shuffle=True,
#     collate_fn=lambda batch: pad(batch)
# )
dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 32,  
    shuffle=True,
    collate_fn=lambda batch: pad(batch)
)
logging.info("Datasets loaded.")


logging.info("Initializing model..")

model = BertSimple(
    device=DEVICE,
    ignore_id=-1,
    num_labels=9, 
    tokenizer=train_dataset.tokenizer,
)

logging.info('Fitting model...')
model.fit(train_loader=train_loader, epochs=10)

logging.info('Evaluating model...')
binary_f1, proportion_f1 = model.evaluate(dev_loader)
logging.info("Binary F1: {}".format(binary_f1))
logging.info("Proportional F1: {}".format(proportion_f1))



################   GRAVEYARD   ################
# model = Transformer(
#     NORBERT='ltgoslo/norbert',
#     tokenizer=train_dataset.tokenizer,
#     num_labels=9, 
#     IGNORE_ID=-1, 
#     device=DEVICE,
#     epochs=2,
#     lr_scheduler=False,
#     factor=0.1,
#     lrs_patience=2,
#     loss_funct='cross-entropy',
#     random_state=1,
#     verbose=True,
#     lr=0.0001,
#     momentum=0.9,
#     epoch_patience=1,
#     label_indexer=None,
#     optimizer='AdamW'
# )
