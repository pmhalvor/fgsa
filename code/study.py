from torch.utils.data import DataLoader
import logging
import torch
import os
import argparse

### LOCAL
from config import DATA_DIR
from config import default_parameters
from config import log_template
from dataset import Norec 
from model import FgsaLSTM
from utils import pad


class Study(): 
    def __init__(self, **kwargs):
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

        set_metric_level = set_metric_level

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

    def store_kwargs(self, kwargs):
        for arg in kwargs:
            self.__dict__[arg] = kwargs[args]

    def load_datasets(self):
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

    def load_dataloaders(self):
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

    def init_model(self, load_from=None):
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

    def fit(self):
        """ 
        scikit-learn like fit() call for use in pipelines and gridsearches
        """  
        logging.info('Fitting model...')
        model.fit(train_loader=train_loader, dev_loader=train_loader, epochs=epochs)

    def score(self, metric="easy"):
        """
        scikit-learn like score() to  use in GridSearchCV

        Parameters:
            metric (str): ["easy", "hard", "strict", "binary", "proportional"]
        """
        logging.info('Evaluating model...')
        absa_f1, easy_f1, hard_f1 = model.evaluate(dev_loader, verbose=True)

        logging.info("ABSA F1: {}".format(absa_f1))
        logging.info("Easy F1: {}".format(easy_f1))
        logging.info("Hard F1: {}".format(hard_f1))

        final = None
        if metric == "easy":
            final = easy_f1
        elif metric == "hard":
            final = hard_f1
        elif metric == "strict":
            final = absa_f1
        elif metric == "binary":
            raise NotImplementedError
            binary = None
            final = binary
        elif metric == "proportional":
            raise NotImplementedError
            proportional = None
            final = proportional

        return final

    def save_model(self):
        logging.info("Saving model to checkpoints/{}.pt".format(name))
        torch.save(model, "/checkpoints/" + name + '.pt')


if __name__ == "__main__":
    hyperparameters = {
        "debug": False,
        "epochs": 50,
        "proportion": 1.,
        "load_checkpoint": False,
        "bert_finetune": False,  # bert frosty
        "subtasks": [
        #    "expression",
        #    "holder",
            "polarity",      # polarity and target only gave results after 10 epochs
            "target", 
        ],

        "learning_rate": 1e-6,
        "lrs": {
            "expression": 1e-7,
            "holder": 1e-6,
            "polarity": 5e-7,
            "target": 5e-7,
        },

        "metric": "hard",  # probably something easier for hyperparameter tuning?

        "loss_function": "cross-entropy",
        "loss_weight": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    name =  "lstm-frostbert-target-polarity-full",
    if proportion<1:
        name += '-{percent}p'.format(
            percent=int(100*proportion)
        )
    if debug:
        name += "-debug"
        level = logging.DEBUG
    else:
        level = logging.INFO

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epoch", type=int, default=0, help="Interactive if overwrite necessary")
    parser.add_argument("-o", "--overwrite", dest="overwrite", type=int, default=1, help="Force overwrite (if not interactive)")
    args = parser.parse_args()

    run(**vars(args))

    print("complete")
    