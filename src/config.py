import logging
import os 
import datetime
import torch


# TODO hide personal info
HOME = "E:/pmhalvor/" if os.path.exists("E:/pmhalvor/") else "/fp/homes01/u01/ec-pmhalvor/"

# TODO download from git automagically for Travis
BERT_PATH = "ltgoslo/norbert2"
DATA_DIR = HOME + "data/norec_fine/"
LOG_DIR = HOME + "nlp/msc/log/" if "E:" in HOME else "/fp/projects01/ec37/log/"
MODEL_DIR = HOME + "nlp/msc/models/" if "E:" in HOME else "/fp/projects01/ec37/models/" 
NOREC_DIR = HOME + "/nlp/msc/norec_fine"


# TODO Use or delete?
default_parameters = {
        "debug": False,
        "epochs": 25,
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


def log_template(level=logging.INFO, name='', job='train'):
    name = '-'+name if name != '' else name

    today = datetime.date.today()
    log_dir = LOG_DIR + today.strftime("%Y-%m-%d")
    try:
        os.mkdir(log_dir)
    except:
        print('Directory {log_dir} exists'.format(log_dir = log_dir))

    logging.basicConfig(
        filename='{log_dir}/{job}{name}.log'.format(log_dir=log_dir, job=job, name=name),
        level=level,
        format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    )
    logger = logging.getLogger(__name__)

    logging.info('---------  new run: {job}{name} ---------'.format(job=job, name=name))
    return logger
