from torch.utils.data import DataLoader
import logging
import torch

## LOCAL
from config import DATA_DIR
from config import log_train
from dataset import NorecTarget 
from model import BertSimple
from utils import pad


####################  config  ####################
log_train(name='BertSimple-targets-large')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info('Running on device {}'.format(DEVICE))
learning_rate = 1e-6
###################################################


# load train/dev/test data so every build has complete result set
logging.info("Loading datasets..")
train_dataset = NorecTarget(
    data_path=DATA_DIR + "train/", 
    ignore_id=-1,
    proportion=0.15,
    )
dev_dataset = NorecTarget(
    data_path=DATA_DIR + "dev/", 
    ignore_id=-1,
    proportion=0.05,
    tokenizer=train_dataset.tokenizer,
    )


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=False,
    collate_fn=lambda batch: pad(batch)
)
dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 32,  
    shuffle=False,
    collate_fn=lambda batch: pad(batch)
)
logging.info("Datasets loaded.")


logging.info("Initializing model..")

model = BertSimple(
    device=DEVICE,
    ignore_id=-1,
    num_labels=5, 
    lr=learning_rate,
    tokenizer=train_dataset.tokenizer,
)

logging.info('Fitting model...')
model.fit(train_loader=train_loader, dev_loader=train_loader, epochs=5)

logging.info('Evaluating model...')
binary_f1, proportion_f1 = model.evaluate(train_loader, verbose=True)
logging.info("Binary F1: {}".format(binary_f1))
logging.info("Proportional F1: {}".format(proportion_f1))

