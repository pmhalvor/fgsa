# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.preprocessing import OurDataset, pad
    from exam.utils.models import Transformer
except:
    from utils.preprocessing import OurDataset, pad
    from utils.models import Transformer

from torch.utils.data import DataLoader
import torch


# NORBERT = 'exam/saga/216'
# NORBERT = "/cluster/shared/nlpl/data/vectors/latest/216"
NORBERT = 'data/216'

# data_dir = "exam/data/"
data_dir = "data/small/"

train_file =f'{data_dir}train.conll'
dev_file =  f'{data_dir}dev.conll'
test_file = f'{data_dir}test.conll'

print('loading data...')
train_dataset = OurDataset(
    data_file=train_file,
    specify_y=None,
    NORBERT_path=NORBERT,
    tokenizer=None
)

# x_ds, y_ds, att_ds = next(iter(train_dataset))
# sentence_tk_ds = train_dataset.tokenizer.convert_ids_to_tokens(x_ds)
# sentence_ds = train_dataset.tokenizer.decode(x_ds)

dev_dataset = OurDataset(
    data_file=dev_file,
    specify_y=None,
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

test_dataset = OurDataset(
    data_file=test_file,
    specify_y=None,
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: pad(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID)
)

# x, y, att = next(iter(train_loader))

dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID)
)

# x1, y1, att1 = next(iter(dev_loader))

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID)
)

# x2, y2, att2 = next(iter(test_loader))

print('initing model...')
model = Transformer(
    NORBERT=NORBERT,
    tokenizer=train_dataset.tokenizer,
    num_labels=5,
    IGNORE_ID=train_dataset.IGNORE_ID,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    lr_scheduler=False,
    factor=0.1,
    lrs_patience=2,
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr=0.0001,
    momentum=0.9,
    epoch_patience=1,
    label_indexer=None,
    optmizer='AdamW'
)

print('fitting model...')
model.fit(train_loader=train_loader, verbose=True, dev_loader=dev_loader)
binary_f1, propor_f1 = model.evaluate(test_loader)
