import torch
import logging
import pytest
import os

## LOCAL
from dataset import NorecOneHot 
import config

DATA_DIR = config.DATA_DIR # TODO hide personal info
LOGGER = logging.getLogger(__name__)
config.log_test(level=logging.INFO, name="test dataset")


def test_data_dir():
    actual_heads = {}
    
    # main dir
    parent = os.listdir(DATA_DIR)
    logging.debug('Parent directory: {}'.format(parent))

    # update actual_heads with top row of files found
    for child in parent:
        split = os.path.join(DATA_DIR, child)
        files = os.listdir(split)
        logging.debug('Annotated files: {}'.format(files))
        actual_heads[child] = set(files)

    expected_heads = {
            'dev': {

                'holder.txt',
                'opinion.txt',
                'sentence.txt',
                'target.txt',
                'target_polarity.txt',
                },
            'test': {

                'holder.txt',
                'opinion.txt',
                'sentence.txt',
                'target.txt',
                'target_polarity.txt',
                },
            'train': {

                'holder.txt',
                'opinion.txt',
                'sentence.txt',
                'target.txt',
                'target_polarity.txt',
                },
        }

    logging.debug('Actual heads: {}'.format(actual_heads))

    assert actual_heads == expected_heads


@pytest.mark.skip(reason="TODO NotImplemented")    
def test_data_rows():
    """
    Should check row count and tops (bottoms too?) of files.
    """
    actual_heads = {}
    actual_count = {}
    
    # main dir
    parent = os.listdir(DATA_DIR)
    logging.info('Parent directory: {}'.format(parent))

    # update actual_heads with top row of files found
    for split in parent:
        files = os.listdir(split)
        logging.info('Annotated files: {}'.format(files))
        for filename in files:
            filepath = os.path.join(split, filename)
            with open(os.path.join(DATA_DIR, filepath)) as f:
                row = f.readline()
            actual_heads[filepath] = row



    assert True



@pytest.mark.skip(reason="TODO fix holder before continuing")
def test_dataset_shape():
    train_dataset = NorecOneHot(data_path=DATA_DIR + "train/", proportion=0.01)

    for i in range(3):
        LOGGER.info(train_dataset.sentence[i])
        LOGGER.info(train_dataset[i][0].shape)
        LOGGER.info(train_dataset[i][1].shape)
        LOGGER.info(train_dataset[i][2].shape)

        # check shapes
        assert train_dataset[i][0].shape == train_dataset[i][1].shape
        assert train_dataset[i][0].shape == train_dataset[i][2].shape


@pytest.mark.skip(reason="TODO fix holder before continuing")
def test_dataset_values():
    train_dataset = NorecOneHot(data_path=DATA_DIR + "train/", proportion=0.01)
    test_dataset = NorecOneHot(data_path=DATA_DIR + "test/", proportion=0.05)
    dev_dataset = NorecOneHot(data_path=DATA_DIR + "dev/", proportion=0.05)

    # for i in range(3):
    #     print('------- Index {} -------'.format(i))
    #     # LOGGER.info(train_dataset[i][0].shape)
    #     # LOGGER.info(train_dataset[i][1].shape)
    #     # LOGGER.info(train_dataset[i][2].shape)
    #     # LOGGER.info(test_dataset[i])

    #     # check shapes
    #     assert train_dataset[i][0].shape == train_dataset[i][1].shape
    #     assert train_dataset[i][0].shape == train_dataset[i][2].shape

    #     # check values
    #     # assert train_dataset[0][i] == torch.Tensor([102., 13553., 25940., 32452., 5859., 5479., 2185., 103.])
    #     # assert test_dataset[0][i] == torch.Tensor([102., 13553., 25940., 32452., 5859., 5479., 2185., 103.])
    #     # assert dev_dataset[0][i] == torch.Tensor([102., 13553., 25940., 32452., 5859., 5479., 2185., 103.])
        
    #     print('------------------------\n')

    assert True