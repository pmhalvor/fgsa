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
def test_BertSimple_fit(strict):
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
        lr=0,
        tokenizer=train_dataset.tokenizer,
    )

    model.fit(train_loader=train_loader, epochs=0)  # don't train

    weights = model.check_weights()

    assert weights is not None
    assert weights.shape == torch.Size([768, 768])
    if strict:
        assert weights[0][0].item() == 0.0304  # very strict
