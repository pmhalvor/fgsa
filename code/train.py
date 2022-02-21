from torch.utils.data import DataLoader
import logging
import torch

## LOCAL
from config import DATA_DIR
from config import log_train
from dataset import Norec
from dataset import NorecOneHot 
from model import BertSimple
from utils import pad


####################  config  ####################
log_train(name='BertSimple-lr-tune')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info('Running on device {}'.format(DEVICE))
###################################################


# load train/dev/test data so every build has complete result set
logging.info("Loading datasets..")
train_dataset = NorecOneHot(
    data_path=DATA_DIR + "train/", 
    ignore_id=-1,
    proportion=0.55,
    )
# test_dataset = NorecOneHot(
#     data_path=DATA_DIR + "test/", 
#     ignore_id=-1,
#     proportion=0.55,
#     tokenizer=train_dataset.tokenizer,
#     )
dev_dataset = NorecOneHot(
    data_path=DATA_DIR + "dev/", 
    ignore_id=-1,
    proportion=0.55,
    tokenizer=train_dataset.tokenizer,
    )


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=False,
    collate_fn=lambda batch: pad(batch)
)
# test_loader = DataLoader(
#     dataset = test_dataset,
#     batch_size = 32,  # for predict to work
#     shuffle=False,
#     collate_fn=lambda batch: pad(batch)
# )
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
    num_labels=9, 
    lr=1e-7,  # 0.00001
    tokenizer=train_dataset.tokenizer,
)

logging.info('Fitting model...')
model.fit(train_loader=train_loader, dev_loader=dev_loader, epochs=5)

logging.info('Evaluating model...')
easy_f1, hard_f1 = model.evaluate(dev_loader, verbose=True)

logging.info("Easy F1: {}".format(easy_f1))
logging.info("Hard F1: {}".format(hard_f1))

