import logging

# TODO download from git automagically
BERT_PATH = "ltgoslo/norbert"
DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"  # TODO hide personal info


def log_test(level=logging.INFO, name='test'):
    """
    Expected to run by pytest. See pytest.ini for config. 
    Gets printed out in terminal/ slurm file. 
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.info('---------  new run: {} ---------'.format(name))
    return logger


def log_train(level=logging.INFO, name=''):
    name = '-'+name if name is not '' else name
    logging.basicConfig(
        filename='../log/train{}.log'.format(name),
        level=level,
        format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    )
    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))  # didnt work as expected
    logging.info('---------  new run: train{} ---------'.format(name))
    return logger


def log_pre(level=logging.INFO, name=''):
    name = '-'+name if name is not '' else name
    logging.basicConfig(
        filename='../log/pre{}.log'.format(name),
        level=level,
        format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    )
    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))  # didnt work as expected
    logging.info('---------  new run: pre{} ---------'.format(name))
    return logger
