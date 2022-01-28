import logging

# TODO download from git automagically
DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"  # TODO hide personal info

def log_test(level=logging.INFO):
    """
    Not in use anymore
    """
    raise NotImplementedError
    logging.basicConfig(
        filename='../log/test.log',
        level=level,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    logging.info('----------------------------------  new test run ----------------------------------')
    return logger


def log_train(level=logging.INFO):
    logging.basicConfig(
        filename='log/out.log',
        level=level,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))  # didnt work as expected
    logging.info('----------------------------------  new run: model.py ----------------------------------')
    return logger
