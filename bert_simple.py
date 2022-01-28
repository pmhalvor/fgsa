# Std. Python imports
import argparse
from tqdm import tqdm

# ML specific imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertModel


"""
WARNING: This may be a non-working version.

See transformer.py for other version.

"""

class BertSimple(nn.Module):
    def __init__(
        self, 
        device,
        bert_path="ltgoslo/norbert",  
        bert_dropout=0.1,       # TODO tune
        bert_finetune = True,   # TODO tune
        learning_rate=0.01,     # TODO tune
        output_dim=5,  # target, holder, expression, polarity, intensity
    ):
        """
        Set up model specific architectures. 

        """
        super(BertSimple, self).__init__()

        self.device = device
        self.dropout = bert_dropout  # TODO potentially refactor name?
        self.learning_rate = learning_rate
        self.output_dim = output_dim

        # initialize contextual embeddings
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert.requires_grad = bert_finetune
        self.bert_dropout = nn.Dropout(self.dropout)

        # ensure everything is on specified device
        self.bert = self.bert.to(self.device)
        self.bert_dropout = self.bert_dropout.to(self.device)  # TODO is this needed?

        # set up output layer
        self.linear = nn.Linear(
            in_features=768,  # size of bert embeddings
            out_features=5
        )
        self.linear = self.linear.to(self.device)

        # optimizer in model for sklearn-style fit() training
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )  # TODO test other optimizers?

    def forward(self, x):
        """
        One forward step of training for our model.

        Parameters:
            x: token ids for a batch
        """

        emb = self.bert(
            x.to(self.device),
            output_hidden_states=True,
        )

        emb = emb.last_hidden_state

        import IPython
        IPython.embed()

    def fit(self, train_loader, dev_loader, epochs=10):
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            num_batches = 0

            loader_iterator = tqdm(train_loader)
            for raw_text, x, y, mask, idx in train_loader:
                self.zero_grad()  # clear updates from prev epoch

                batches_len, seq_len = x.shape

                preds = self.forward(
                    x.to(self.device)  # FIXME does this need to happen everywhere? 
                )

                # TODO continue dev when this has been checked




