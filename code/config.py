import logging
import os 
import datetime

# TODO download from git automagically
BERT_PATH = "ltgoslo/norbert"
DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"  # TODO hide personal info


def log_template(level=logging.INFO, name='', job='train'):
    name = '-'+name if name != '' else name

    today = datetime.date.today()
    log_dir = '../log/' + today.strftime("%Y-%m-%d")
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
