import logging
import os 
import datetime
import torch


HOME = "E:/pmhalvor/nlp/msc/" if os.path.exists("E:/pmhalvor/nlp/msc/") else os.getcwd()
HOME += "/../" if "fgsa" in HOME else "/"


BERT_PATH = "ltgoslo/norbert2"
DATA_DIR = HOME + "data/norec_fine/"
LOG_DIR = HOME + "log/" 
MODEL_DIR = HOME + "models/" 
NOREC_DIR = HOME + "norec_fine"


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
