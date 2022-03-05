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
    """
    Expects DATA_DIR to be defined in config.py as the path to IMN-style data.

    Note: only pass model specific parameters as kwargs.
    """ 
    def __init__(
        self, 
        name="base",

        bert_finetune=False,
        batch_size=32, 
        data_dir=DATA_DIR,
        debug = False, 
        epochs = 50,
        ignore_id = -1,
        load_checkpoint = False,

        learning_rate = 1e-7,
        lrs = None,

        metric = "easy",

        model_name = "FgsaLSTM",
        model_path = None,
        proportion = 1.,
        shuffle = True,
        subtasks = [
           "expression",
           "holder",
            "polarity",
            "target", 
        ],

        loss_function = "cross-entropy",

        verbose = False,
        **kwargs
    ):
        self.bert_finetune = bert_finetune
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.debug = debug
        self.epochs = epochs
        self.ignore_id = ignore_id
        self.kwargs = kwargs
        self.load_checkpoint = load_checkpoint
        
        self.learning_rate = learning_rate
        self.lrs = lrs

        self.loss_function = loss_function

        self.metric = metric

        self.model_name = model_name
        self.model_path = model_path  # TODO refactor into model

        self.proportion = proportion
        self.shuffle = shuffle
        self.subtasks = subtasks

        self.name = name
        self.build_name()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = self.build_logger()
        self.logger.info('Running on device {}'.format(self.device))
        self.verbose = verbose

        ###### load data ######
        self.load_datasets()
        self.load_dataloaders()

        ###### init model #####
        self.init_model(model_path=model_path)
        self.logger.info("Ready to fit!")

    def store_kwargs(self, kwargs):
        for arg in kwargs:
            self.__dict__[arg] = kwargs[args]

    def find(self, arg):
        return self.__dict__.get(arg)

    def build_name(self, *args):
        if args:
            for arg in args:
                self.name += "-" + arg
        else:
            if self.proportion<1:
                self.name += '-{percent}p'.format(
                    percent=int(100*self.proportion)
                )
            if self.debug:
                self.name += "-debug"
        
    def build_logger(self):
        if self.debug:
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO
        return log_template(level=self.level, job='dev', name=self.name)

    def load_datasets(self):
        """
        Adds attributes self.train_dataset, self.test_dataset, self.dev_dataset
        to this object 
        """
        # load train/dev/test data so every build has complete result set
        self.logger.info("Loading datasets..")
        self.train_dataset = Norec(
            data_dir=self.data_dir,
            partition="train/", 
            ignore_id=self.ignore_id,
            proportion=self.proportion,
        )
        self.test_dataset = Norec(
            data_dir=self.data_dir,
            partition="test/", 
            ignore_id=self.ignore_id,
            proportion=self.proportion,
            tokenizer=self.train_dataset.tokenizer,
        )
        self.dev_dataset = Norec(
            data_dir=self.data_dir,
            partition="dev/", 
            ignore_id=self.ignore_id,
            proportion=self.proportion,
            tokenizer=self.train_dataset.tokenizer,
        )

    def load_dataloaders(self):
        """ 
        Adds attributes train_loader, test_loader, dev_loader to this object
        """
        if self.find("train_dataset") is None:
            self.load_datasets()

        # data loader
        self.train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle=self.shuffle,
            collate_fn=lambda batch: pad(batch)
        )
        self.test_loader = DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            shuffle=self.shuffle,
            collate_fn=lambda batch: pad(batch)
        )
        self.dev_loader = DataLoader(
            dataset = self.dev_dataset,
            batch_size = self.batch_size,
            shuffle=self.shuffle,
            collate_fn=lambda batch: pad(batch)
        )
        self.logger.info("Datasets loaded.")

    def init_model(self, model_path=None):
        """
        Create a model attribute for this object
        """
        self.logger.info("Initializing model..")
        if self.load_checkpoint and os.path.exists("/checkpoints/{}.pt".format(self.name)):
            self.model = torch.load("/checkpoints/" + self.name + '.pt', map_location=torch.device(self.device))
            self.logger.info("... from checkpoint/{}.pt".format(self.name))
        elif model_path is not None and os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=torch.device(self.device))
            self.logger.info("... from {}".format(model_path))
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model = torch.load(self.model_path, map_location=torch.device(self.device))
            self.logging.info("... from {}".format(self.model_path))
        else:
            model_class = self.get_model_class(self.model_name)
            self.model = model_class(**self.params())
            self.logger.info("... from new {} object.".format(self.model_name))

    def get_model_class(self, model_name=None):
        model = None
        
        if model_name.lower() == "fgsalstm":
            model = FgsaLSTM
        elif model_name.lower() == "berthead":
            model = BertHead
        # elif if model_name.lower() == "next":
        #   model = Next

        return model

    def params(self):
        params = {
            "bert_finetune": self.bert_finetune,
            "debug": self.debug,
            "device": self.device,
            "epochs": self.epochs,
            "ignore_id": self.ignore_id, 
            "learning_rate": self.learning_rate,
            "lrs": self.lrs,
            "loss_function": self.loss_function,
            "subtasks": self.subtasks,
        }
        params.update(self.kwargs)

        print("Current params: {}".format(params))
        self.logger.info("Current params: {}".format(params))

        # to avoid messy logs
        params["tokenizer"] = self.train_dataset.tokenizer
        return params

    def fit(self):
        """ 
        scikit-learn like fit() call for use in pipelines and gridsearches.

        TODO learn if you need to feed data in through grid search.
        """  
        self.logger.info('Fitting model...')
        self.model.fit(
            train_loader=self.train_loader, 
            dev_loader=self.dev_loader, 
            epochs=self.epochs
        )

    def score(self, metric=None):
        """
        scikit-learn like score() to  use in GridSearchCV

        Parameters:
            metric (str): ["easy", "hard", "strict", "binary", "proportional"]
        """
        self.logger.info('Scoring model...')
        absa_f1, easy_f1, hard_f1 = self.model.evaluate(self.dev_loader, verbose=self.verbose)

        self.logger.info("ABSA F1: {}".format(absa_f1))
        self.logger.info("Easy F1: {}".format(easy_f1))
        self.logger.info("Hard F1: {}".format(hard_f1))

        if metric is None:
            metric = self.metric 
        self.final = None
        if metric == "easy":
            self.final = easy_f1
        elif metric == "hard":
            self.final = hard_f1
        elif metric == "strict":
            self.final = absa_f1
        elif metric == "binary":
            raise NotImplementedError
            binary = None
            self.final = binary
        elif metric == "proportional":
            raise NotImplementedError
            proportional = None
            self.final = proportional

        return self.final

    def save_model(self):
        logging.info("Saving model to checkpoints/{}.pt".format(name))
        torch.save(model, "/checkpoints/" + name + '.pt')


if __name__ == "__main__":
    import logging
    import json
    import sys

    if len(sys.argv) > 0:
        filename = sys.argv[1]
        with open('../studies/'+filename, 'r') as f:  # find sutdy configs in studies/ dir
            data = json.load(f)

        params = {k : data[k][0] if isinstance(data[k], list) else data[k] for k in data}
        if data.get("subtasks"):
            params["subtasks"] = data["subtasks"].copy()

        for param in data:
            if isinstance(data[param], list) and param != "subtasks":
                print("Looking into {}".format(param))

                # null best values
                best_hyper = None
                best_score = -1. 

                # hyperparameter search
                for hyper in data[param]:
                    params.pop(param)
                    params[param] = hyper
                    study = Study(**params)
                    study.fit()
                    results = study.score()
                    print("Results for {p}={h}: {r}".format(p=param, h=hyper, r=results))

                    if results > best_score:
                        best_hyper = hyper
                
                params[param] = best_hyper
                print("Best results for {m} metric: {p}={h}".format(m=study.metric, p=param, h=best_hyper))
