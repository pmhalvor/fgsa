from torch.utils.data import DataLoader
import torch

## LOCAL 
from config import DATA_DIR
from config import log_test
from dataset import NorecOneHot
from model import BertSimple
from utils import pad

log_test()

def test_BertSimple_fit():
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
        lr=1e-7,  # 0.00001
        tokenizer=train_dataset.tokenizer,
    )

    
    model.fit(train_loader=train_loader, epochs=1)

    weights = model.check_weights()

    assert weights is not None
    logging.info(weights)
