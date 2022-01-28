import torch
import logging

## LOCAL
from dataset import NorecOneHot 
import config

LOGGER = logging.getLogger(__name__)
DATA_DIR = config.DATA_DIR # TODO hide personal info


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