import logging

# TODO download from git automagically
BERT_PATH = "ltgoslo/norbert"
DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"  # TODO hide personal info


# def log_test(level=logging.INFO, name='test'):
#     """
#     Expected to run by pytest files to  See pytest.ini for config. 
#     Gets printed out in terminal/ slurm file. 
#     """
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level)
#     logger.info('---------  new test: {} ---------'.format(name))
#     return logger


def log_template(level=logging.INFO, name='', job='train'):
    name = '-'+name if name is not '' else name

    logging.basicConfig(
        filename='../log/{job}{name}.log'.format(job=job, name=name),
        level=level,
        format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    )
    logger = logging.getLogger(__name__)

    logging.info('---------  new run: {job}{name} ---------'.format(job=job, name=name))
    return logger
