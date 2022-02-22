import torch
import logging
import pytest
import os

## LOCAL
from dataset import NorecOneHot 
from utils import compare
import config

DATA_DIR = config.DATA_DIR # TODO hide personal info


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


def test_dataset_shape():
    train_dataset = NorecOneHot(data_path=DATA_DIR + "train/", proportion=0.05)

    for i in range(3):
        logging.debug(train_dataset.sentence[i])
        logging.debug(train_dataset[i][0].shape)
        logging.debug(train_dataset[i][1].shape)
        logging.debug(train_dataset[i][2].shape)

        # check shapes
        assert train_dataset[i][0].shape == train_dataset[i][1].shape
        assert train_dataset[i][0].shape == train_dataset[i][2].shape


def test_dataset_values():
    train_dataset = NorecOneHot(data_path=DATA_DIR + "train/", proportion=0.01)
    test_dataset = NorecOneHot(data_path=DATA_DIR + "test/", proportion=0.05)
    dev_dataset = NorecOneHot(data_path=DATA_DIR + "dev/", proportion=0.05)

    # check shapes
    assert train_dataset[0][0].shape == train_dataset[0][1].shape
    assert train_dataset[0][0].shape == train_dataset[0][2].shape

    # check values
    assert compare(train_dataset[0][0], torch.Tensor([ 101, 1915, 5859, 5479, 2185]))
    assert compare(test_dataset[0][0], torch.Tensor([ 4994,  3060,  8464,  3357,  8611, 32293, 31757]))
    assert compare(dev_dataset[0][0], torch.Tensor([ 3665, 23507, 31804,   101]))

    assert compare(train_dataset[0][1], torch.Tensor([1, 1, 1, 1, 1]))
    assert compare(test_dataset[0][1], torch.Tensor([1, 1, 1, 1, 1, 1, 1]))
    assert compare(dev_dataset[0][1], torch.Tensor([1, 1, 1, 1]))

    assert compare(train_dataset[0][2], torch.Tensor([0, 0, 0, 0, 0]))
    assert compare(test_dataset[0][2], torch.Tensor([0, 0, 0, 0, 0, 0, 0]))
    assert compare(dev_dataset[0][2], torch.Tensor([0, 0, 0, 0]))

    assert True

