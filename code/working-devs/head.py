from torch.utils.data import DataLoader
import logging
import torch

## LOCAL
from config import DATA_DIR
from config import log_template
from dataset import Norec 
from model import BertHead
from utils import pad


####################  config  ####################
debug = False 
epochs = 200
label_importance = 10
learning_rate = 1e-6
proportion = 0.5
load_checkpoint = False

name = "head"
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
if load_checkpoint:
    try:
        model = torch.load("/checkpoints/" + name + '.pt', map_location=torch.device(DEVICE))
        logging.info("... from checkpoint/{}.pt".format(name))
    except FileNotFoundError:
        model = BertHead(
            device=DEVICE,
            ignore_id=-1,
            lr=learning_rate,
            tokenizer=train_dataset.tokenizer,
            label_importance=label_importance,
        ) 
        logging.info("... from new instance.")
else:
    model = BertHead(
        device=DEVICE,
        ignore_id=-1,
        lr=learning_rate,
        tokenizer=train_dataset.tokenizer,
        label_importance=label_importance,
    )
    logging.info("... from new instance.")

logging.info('Fitting model...')
model.fit(train_loader=train_loader, dev_loader=train_loader, epochs=epochs)

logging.info('Evaluating model...')
easy_f1, hard_f1 = model.evaluate(dev_loader, verbose=True)

logging.info("Easy F1: {}".format(easy_f1))
logging.info("Hard F1: {}".format(hard_f1))

logging.info("Saving model to checkpoints/{}.pt".format(name))
torch.save(model, "/checkpoints/" + name + '.pt')

