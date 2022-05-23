from torch.utils.data import DataLoader
import logging
import torch

## LOCAL
from config import DATA_DIR
from config import log_template
from dataset import Norec
from model import FgFlex
from utils import pad


####################  config  ####################
parameters = {
    "name": "single-trained-FgFlex",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "attention_relations": [
        ["target", "expression"], 
        ["target", "holder"], 

        ["expression", "target"], 
        ["expression", "holder"], 

        ["holder", "target"], 
        ["holder", "expression"], 

        ["polarity", "target"], 
        ["polarity", "holder"], 
        ["polarity", "expression"]
    ],     
    "cnn_dim": 768,
    "expanding_cnn": 0,
    "gold_transmission": True,
    "kernel_size": 5,
    "learning_rate": 1e-5,
    "stack_count": 2,

    "shared_layers": 3,
    "expression_layers": 3,
    "holder_layers": 2,
    "polarity_layers": 1,
    "target_layers": 3,
}

logging.info('Running on device {}'.format(parameters["device"]))
###################################################


# load train/dev/test data so every build has complete result set
logging.info("Loading datasets..")
train_dataset = Norec(
    data_dir=DATA_DIR,
    proportion=1.,
    partition="train",
    )
test_dataset = Norec(
    data_dir=DATA_DIR, 
    ignore_id=-1,
    partition="test",
    proportion=1.,
    tokenizer=train_dataset.tokenizer,
    )
dev_dataset = Norec(
    data_dir=DATA_DIR, 
    ignore_id=-1,
    partition="dev",
    proportion=1.,
    tokenizer=train_dataset.tokenizer,
    )


# data loader
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle=False,
    collate_fn=lambda batch: pad(batch)
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 32,  # for predict to work
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

logging.info("Loading new FgFlex model with the following parameters...")
logging.info(parameters)
model = FgFlex(
    tokenizer=train_dataset.tokenizer,
    **parameters,
)

logging.info('Fitting model...')
print("Fitting model")
model.fit(train_loader=train_loader, dev_loader=dev_loader, epochs=5)

logging.info('Evaluating model...')
absa, binary, hard, macro, proportional, span = model.evaluate(dev_loader, verbose=True)

logging.info("  ABSA F1: {}".format(absa))
logging.info("Binary F1: {}".format(binary))
logging.info("  Hard F1: {}".format(hard))
logging.info(" Macro F1: {}".format(Macro))
logging.info(" Prop. F1: {}".format(proportional))
logging.info("  Span F1: {}".format(span))

logging.info("Saving model to {}".format("/checkpoints/" + parameters["name"] + '.pt'))
torch.save(model, "/checkpoints/" + name + '.pt')

