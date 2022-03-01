from torch.utils.data import DataLoader
import logging
import torch
import os

### LOCAL
from config import DATA_DIR
from config import log_template
from dataset import Norec 
from model import FgsaLSTM
from utils import pad


####################  config  ####################
debug = False 
epochs = 50
proportion = 1.
load_checkpoint = False
bert_finetune=False  # bert frosty
subtasks = [
#    "expression",
#    "holder",
    "polarity",      # polarity and target only gave results after 10 epochs
    "target", 
]

learning_rate = 1e-6
lrs = {
    "expression": 1e-7,
    "holder": 1e-6,
    "polarity": 5e-7,
    "target": 5e-7,
}

loss_function = "cross-entropy"
loss_weight = 3

name = "lstm-frostbert-target-polarity-full"
if proportion<1:
    name += '-{percent}p'.format(
        percent=int(100*proportion)
    )
if debug:
    name += "-debug"
    level = logging.DEBUG
else:
    level = logging.INFO
log_template(level=level, job='dev', name=name)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info('Running on device {}'.format(DEVICE))
###################################################


# load train/dev/test data so every build has complete result set
logging.info("Loading datasets..")
train_dataset = Norec(
    data_dir=DATA_DIR,
    partition="train/", 
    ignore_id=-1,
    proportion=proportion,
)
dev_dataset = Norec(
    data_dir=DATA_DIR,
    partition="dev/", 
    ignore_id=-1,
    proportion=proportion,
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
if load_checkpoint and os.path.exists("/checkpoints/" + name + '.pt'):
    model = torch.load("/checkpoints/" + name + '.pt', map_location=torch.device(DEVICE))
    logging.info("... from checkpoint/{}.pt".format(name))
else:
    model = FgsaLSTM(
        device=DEVICE,
        bert_finetune=bert_finetune,
        ignore_id=-1,
        loss_function=loss_function,
        loss_weight=loss_weight,
        lr=learning_rate,
        tokenizer=train_dataset.tokenizer,
        subtasks=subtasks,
        expression_lr=lrs.get("expression"),  # dict.get() defaults to None
        holder_lr=lrs.get("holder"),
        polarity_lr=lrs.get("polarity"),
        target_lr=lrs.get("target"),
    )
    logging.info("... from new instance.")


logging.info('Fitting model...')
model.fit(train_loader=train_loader, dev_loader=train_loader, epochs=epochs)


logging.info('Evaluating model...')
absa_f1, easy_f1, hard_f1 = model.evaluate(dev_loader, verbose=True)

logging.info("ABSA F1: {}".format(absa_f1))
logging.info("Easy F1: {}".format(easy_f1))
logging.info("Hard F1: {}".format(hard_f1))


logging.info("Saving model to checkpoints/{}.pt".format(name))
torch.save(model, "/checkpoints/" + name + '.pt')

