import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader
from transformers import BertModel
import numpy as np

from utils.metrics import binary_analysis, proportional_analysis, get_analysis

import argparse
from tqdm import tqdm


class BertBiLSTM(nn.Module):

    def __init__(
        self, 
        vocab_size,
        bert_path,
        embedding_dim,
        hidden_dim,
        device,
        output_dim=5,
        num_layers=2,
        lstm_dropout=0.2,
        word_dropout=0.5,
        learning_rate=0.01,
        train_embeddings=False,
        NORBERT=None,
        verbose=False,
    ):
        super(BertBiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.learning_rate = learning_rate
        self.sentiment_criterion = nn.CrossEntropyLoss()
        self.verbose = verbose

        # set up pretrained embeddings
        self.word_embeds = BertModel.from_pretrained(bert_path)
        self.word_embeds.to(self.device)
        self.word_embeds.requires_grad = train_embeddings

        self.word_dropout = nn.Dropout(word_dropout)

        # make sure model params on correct device
        self.word_embeds = self.word_embeds.to(self.device)
        self.word_dropout = self.word_dropout.to(self.device)

        # set up BiLSTM and linear layers
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True).to(self.device)

        self.linear = nn.Linear(hidden_dim * 2, self.output_dim).to(self.device)

        # We include the optimizer here to enable sklearn style fit() method
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

    def init_hidden(self, batch_size=1):
        """
        :param batch_size: batch size for the training/dev/test batch
        """
        h0 = torch.zeros((self.lstm.num_layers * (1 + self.lstm.bidirectional),
                          batch_size, self.lstm.hidden_size)).to(self.device)
        c0 = torch.zeros_like(h0).to(self.device)
        return (h0, c0)

    def forward(self, x, mask):
        """
        :param x: a packed padded sequence
        """
        # get the batch sizes, which will be used for packing embeddings
        # batch_size = x.batch_sizes[0]

        # Embed and add dropout
        emb = self.word_embeds(
            x.to(self.device),
            attention_mask=mask.to(self.device),
            output_hidden_states=True
        )
        emb = emb.last_hidden_state # TODO check if this is right? potiential BUG
        emb = self.word_dropout(emb)

        # Pack and pass to LSTM layer
        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(
            input=emb, # final state of Bert embedding representation
            lengths=mask.sum(dim=1).to(self.device), # attention_mask
            batch_first=True,
            enforce_sorted=False
        )
        self.hidden = self.init_hidden(x.shape[0])
        output, (hn, cn) = self.lstm(packed_emb, self.hidden)

        # Unpack and send to linear layer
        o, _ = pad_packed_sequence(output, batch_first=True)
        o = self.linear(o)
        return o

    def fit(self, train_loader, dev_loader, epochs=10):
        """
        Trains a model in an Sklearn style manner.
        :param dev_loader: torch.utils.data.DataLoader object of the train data
        :param dev_loader: torch.utils.data.DataLoader object of the dev data
                           with batch_size=1
        :param epochs: number of epochs to train the model
        """

        for epoch in range(epochs):
            # Iterate over training data
            self.train()
            epoch_loss = 0
            num_batches = 0

            # Data reader requires each example to have (raw_text, x, y, idx)
            loader_iterator = tqdm(train_loader) if self.verbose else train_loader
            for raw_text, x, y, mask, idx in train_loader:
                self.zero_grad() # make sure previous epoch upadtes are cleared

                # Get the batch_sizes, batches_len, and seq_length for future
                # changes to data view
                # original_data, batch_sizes = pad_packed_sequence(x, batch_first=True)
                batches_len, seq_length = x.shape

                # Get the predictions and the gold labels (y)
                preds = self.forward(
                    x.to(self.device),
                    mask.to(self.device)
                )

                # Reshape predictions (batch_size * max_seq_length, num_labels)
                # i.e. flattens batch to single dim
                preds = preds.reshape(batches_len * seq_length, 5)  # BUG? readd to(device)
                y = y.reshape(batches_len * seq_length) # BUG? readd to(device)


                # Get loss and update epoch_loss
                loss = self.sentiment_criterion(preds, y.to(self.device))
                epoch_loss += loss.data
                num_batches += 1

                # Update parameters
                loss.backward()
                self.optimizer.step()

            print()
            print("Epoch {0} loss: {1:.3f}".format(epoch + 1,
                                                   epoch_loss / num_batches))

            print("Dev")
            self.evaluate(dev_loader)

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions = []
        golds = []
        sents = []
        for raw_text, x, y, attention_mask, idx in tqdm(test_loader):
            preds = self.forward(
                x,
                attention_mask
            ).argmax(2)
            predictions.append(preds[0])
            golds.append(y.data)
            sents.append(raw_text[0])
        return predictions, golds, sents

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the examples passed via test_loader.
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        preds, golds, sents = self.predict(test_loader)
        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [i.item() for l in golds for i in l]

        analysis = get_analysis(sents, preds, golds)
        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    def print_predictions(self, test_loader, outfile, idx2label):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        :param outfile: the file name to print the predictions to
        :param idx2label: a python dictionary which maps label indices to
                          the actual label
        """
        preds, golds, sents = self.predict(test_loader)
        with open(outfile, "w") as out:
            for sent, gold, pred in zip(sents, golds, preds):
                for token, gl, pl in zip(sent, gold, pred):
                    glabel = idx2label[int(gl)]
                    plabel = idx2label[int(pl)]
                    out.write(("{0}\t{1}\t{2}\n".format(token,
                                                        glabel,
                                                        plabel)))
                out.write("\n")

