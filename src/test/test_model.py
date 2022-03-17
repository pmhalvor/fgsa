from torch.utils.data import DataLoader
import torch
import logging
import pytest

## LOCAL 
from config import DATA_DIR
from dataset import NorecOneHot
from model import BertSimple
from utils import pad

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info('Running on device {}'.format(DEVICE))

strictness = [True, False]


@pytest.mark.parametrize("strict", strictness)
def test_BertSimple(strict):
    train_dataset = NorecOneHot(
        data_path=DATA_DIR + "train/", 
        ignore_id=-1,
        proportion=0.01,
    )

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = 32,
        shuffle=False,
        collate_fn=lambda batch: pad(batch)
    )

    model = BertSimple(
        device=DEVICE,
        ignore_id=-1,
        num_labels=9, 
        lr=1e-6,
        tokenizer=train_dataset.tokenizer,
    )


    weights_before = model.check_weights()

    model.fit(train_loader=train_loader, epochs=1)

    weights_after = model.check_weights()

    assert weights_after is not None
    assert weights_before != weights_after  # model has "learned" something
    assert weights_before.shape == torch.Size([768, 768])
    assert weights_after.shape == torch.Size([768, 768])
    if strict:
        assert weights_before[0][0].item() == 0.0304  # very strict

    # TODO Add shape check for outputs
    # TODO Add evaluation checks 
