from tqdm import tqdm
from itertools import chain  # for Transformer
import argparse
import numpy as np  # for Transformer
import logging

## ML specific
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence  # for Transformer
import torch
import torch.nn as nn

from transformers import BertForTokenClassification
from transformers import BertModel
from transformers import BertTokenizer  # for Transformer

## Local imports
from metrics import binary_analysis  # for Transformer
from metrics import proportional_analysis  # for Transformer
from metrics import get_analysis  # for Transformer
from utils import decode_batch
from utils import score

class BertSimple(nn.Module):
    """
    Built specifically for fgsa code
    """
    def __init__(
        self, 
        device,
        num_labels,
        bert_path="ltgoslo/norbert",  
        bert_dropout=0.1,           # TODO tune
        bert_finetune=True,         # TODO tune
        ignore_id=-1,
        lr=0.01,                    # TODO tune
        lr_scheduler_factor=0.1,    # TODO tune
        lr_scheduler_patience=2,    # TODO tune
        output_dim=5,  # target, holder, expression, polarity, intensity
        tokenizer=None,
    ):
        """
        Set up model specific architectures. 

        """
        super(BertSimple, self).__init__()

        self.device = device
        self.dropout = bert_dropout  # TODO potentially refactor name?
        self.finetune = bert_finetune
        self.learning_rate = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.output_dim = output_dim
        self.tokenizer = tokenizer

        # initialize contextual embeddings
        self.bert = BertForTokenClassification.from_pretrained(
            bert_path, num_labels=num_labels
        )
        self.bert.requires_grad = self.finetune
        self.bert_dropout = nn.Dropout(self.dropout)

        # ensure everything is on specified device
        self.bert = self.bert.to(self.device)
        self.bert_dropout = self.bert_dropout.to(self.device)  # TODO is this needed?

        # loss function
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_id)

        # optimizer in model for sklearn-style fit() training
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )  # TODO test other optimizers?

        # setting learning rate's scheduler
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience
        )


    def fit(self, train_loader, epochs=10):
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            num_batches = 0

            loader_iterator = tqdm(train_loader)
            for b, batch in enumerate(train_loader):
                self.train()        # turn off eval mode
                self.zero_grad()    # clear updates from prev epoch

                outputs = self.forward(batch)
                
                targets = batch[2]

                # if epoch<3 and b<3:
                #     # # logging.info("Keys in output dict: {}".format(outputs.__dict__.keys()))
                #     # # logging.info("batch[0][:3]: {}".format(batch[0][:3]))
                #     # logging.info("batch[0].shape: {}".format(batch[0].shape))
                #     # # logging.info("batch[1][:3]: {}".format(batch[1][:3]))
                #     # logging.info("batch[1].shape: {}".format(batch[1].shape))
                #     # # logging.info("target[:3]: {}".format(targets[:3]))
                #     # logging.info("target shape: {}".format(targets.shape))
                #     # logging.info("logits shape: {}".format(outputs.logits.shape))
                #     # logging.info("logits premuted: {}".format(outputs.logits.permute(0, 2, 1).shape))
                #     logging.info("loss: {}".format(outputs.loss))
                
                # apply loss
                loss = self.backward(outputs.logits.permute(0, 2, 1), targets)
                logging.info("Epoch:{:3} Batch:{:3} Loss:{}".format(epoch, b, loss.item()))
        logging.info("Fit complete.")

    
    def forward(self, batch):
        """
        One forward step of training for our model.

        Parameters:
            x: token ids for a batch
        """
        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))
        labels = batch[2].to(torch.device(self.device))

        return self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            labels = labels
        )


    def backward(self, outputs, targets):
        """
        Performs a backpropagation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The real targets.
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        # self.check_weights()

        computed_loss = self.loss(
            input=outputs.to(torch.device(self.device)),
            target=targets.to(torch.device(self.device))  # FIXME where is this supposed to happen?
            )

        # calculating gradients
        computed_loss.backward()

        # self.check_weights()

        # updating weights from the model by calling optimizer.step()
        self.optimizer.step()

        # self.check_weights()

        # quit()
        return computed_loss


    def evaluate(self, loader):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via loader.

        :param loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        f_absa_overall = 0

        for b, batch in enumerate(loader):
            predictions = self.predict(batch)

            # decode predictions and batch[2]
            predict_decoded = decode_batch(predictions, mask=batch[1])
            true_decoded = decode_batch(batch[2], mask=batch[1])

            annotations = ['expression', 'holder', 'polarity', 'target']

            f_target, acc_polarity, f_polarity, f_absa = score(
                true_aspect = true_decoded["targets"], 
                predict_aspect = predict_decoded["targets"], 
                true_sentiment = true_decoded["polarities"], 
                predict_sentiment = predict_decoded["polarities"], 
                train_op = False
            )

            f_expression, _, _, _ = score(
                true_aspect = true_decoded["expressions"], 
                predict_aspect = predict_decoded["expressions"], 
                true_sentiment = true_decoded["polarities"], 
                predict_sentiment = predict_decoded["polarities"], 
                train_op = True
            )

            f_holder, _, _, _ = score(
                true_aspect = true_decoded["holders"], 
                predict_aspect = predict_decoded["holders"], 
                true_sentiment = true_decoded["polarities"], 
                predict_sentiment = predict_decoded["polarities"], 
                train_op = True
            )

            logging.info("f_target: {}".format(f_target))
            logging.info("f_expression: {}".format(f_expression))
            logging.info("f_holder: {}".format(f_holder))
            logging.info("acc_polarity: {}".format(acc_polarity))
            logging.info("f_polarity: {}".format(f_polarity))
            logging.info("f_absa: {}".format(f_absa))

            f_absa_overall = (f_absa + f_absa_overall)/2.


        return f_absa_overall, None


    def predict(self, batch):
        """
        :param batch: tensor containing batch of dev/test data 
        """
        self.eval()

        outputs = self.forward(batch)

        self.predictions = outputs.logits.argmax(2)

        return self.predictions
        

    def check_weights(self):
        for parent, module in self.bert.named_children():
            if parent == "bert":
                for child, mod in module.named_children():
                    if child=="encoder":
                        for layer, md in mod.named_children():
                            if layer=="layer":
                                for name, bert_layer in md.named_children():
                                    if name=="0":
                                        for att, wrapper in bert_layer.named_children():
                                            if att=="attention":
                                                for s, s_wrap in wrapper.named_children():
                                                    if s=="self":
                                                        for n, m in s_wrap.named_children():
                                                            logging.info("Name:{}  Module:{}  Weight:{}".format(n, m, m.weight))
                                                            break
                                                        break
                                                break
                                        break
                                break
                        break
                

class Transformer(torch.nn.Module):
    """
    Taken from my solution to IN5550 exam.
    """

    @staticmethod
    def _lossFunc(lf_type, IGNORE_ID):
        """
        Returns the specified loss function from torch.nn
        ______________________________________________________________
        Parameters:
        lf_type: str
            The loss function to return
        ______________________________________________________________
        Returns:
        lf: torch.nn.function
            The specified loss function
        """
        if lf_type == "cross-entropy":  # I:(N,C) O:(N)
            return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)

        if lf_type == 'binary-cross-entropy':
            return torch.nn.BCELoss()

    def __init__(
        self,
        NORBERT,
        tokenizer,
        num_labels,
        IGNORE_ID,
        device='cpu',
        epochs=10,
        lr_scheduler=False,
        factor=0.1,
        lrs_patience=2,
        loss_funct='cross-entropy',
        random_state=None,
        verbose=False,
        lr=2e-5,
        momentum=0.9,
        epoch_patience=1,
        label_indexer=None,
        optmizer='SGD'
    ):

        super().__init__()

        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # global parameters
        self.NORBERT = NORBERT
        self.num_labels = num_labels
        self.IGNORE_ID = IGNORE_ID
        self.device = device
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = lrs_patience
        self.epoch_patience = epoch_patience
        self.loss_funct_str = loss_funct
        self._loss_funct = self._lossFunc(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )
        self.lr = lr
        self.momentum = momentum
        self.last_epoch = None

        # setting model
        self.tokenizer = tokenizer
        self.model = BertForTokenClassification.from_pretrained(
            self.NORBERT,
            num_labels=self.num_labels,
        ).to(torch.device(self.device))

        # setting model's optimizer
        self.optmizer = optmizer
        if self.optmizer == 'SGD':
            self._opt = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.optmizer == 'AdamW':
            self._opt = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
            )

        # setting learning rate's scheduler
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self._opt,
            mode='min',
            factor=self.factor,
            patience=self.patience
        )

        # storing scores
        self.losses = []
        self.binary_f1 = []
        self.propor_f1 = []

        # early stop
        self.early_stop_epoch = None

        # label indexer
        self.label_indexer = label_indexer

        # storing outputs
        self.outputs = None

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor

        """
        return self.model(
            input_ids=batch[0].to(torch.device(self.device)),
            attention_mask=batch[2].to(torch.device(self.device)),
            output_hidden_states=True
        )

    def backward(self, outputs, targets):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The real targets.
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        computed_loss = self._loss_funct(
            input=outputs,
            target=targets.to(torch.device(self.device))
            )

        # calculating gradients
        computed_loss.backward()

        # updating weights from the model by calling optimizer.step()
        self._opt.step()

        return computed_loss

    def fit(self, train_loader=None, verbose=False, dev_loader=None):
        """
        Fits the model to the training data using the models
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        laoder: torch.nn.Dataloader=None
            Dataloader object to load the batches, defaults to None
        verbose: bool=False
            If True: prints out progressive output, defaults to False
        ______________________________________________________________
        Returns:
        None
        """
        iterator = range(self.epochs)  # tqdm(range(self.epochs)) if False else 

        for epoch in iterator:
            _loss = []

            for b, batch in enumerate(train_loader):
                self.train()
                self.outputs = self.forward(batch=batch)
                loss = self.backward(
                    outputs=self.outputs.logits.permute(0, 2, 1),
                    targets=batch[1]
                )
                _loss.append(loss.item())

                logging.info("Epoch:{} \t Batch:{} \t Loss:{}".format(epoch, b, loss.item()))
                print("Epoch:{} \t Batch:{} \t Loss:{}".format(epoch, b, loss.item()))

            if self._early_stop(epoch_idx=epoch,
                                patience=self.epoch_patience):
                print('Early stopped!')

                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)
                break

            else:
                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)

        self.last_epoch = epoch
        return self

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        self.predictions, self.golds, self.sents = [], [], []

        for batch in test_loader:  # removed tqdm
            out = self.forward(batch)
            y_pred = out.logits.argmax(2)
            self.predictions.append(y_pred.squeeze(0).tolist())
            self.golds.append(batch[1].squeeze(0).tolist())

            for i in batch[0]:
                self.decoded_sentence = \
                    self.tokenizer.convert_ids_to_tokens(i)
                self.sents.append(self.decoded_sentence)

        # #################### truncating predictions, golds and sents
        self.predictions__, self.golds__, self.sents__ = [], [], []
        for l_p, l_g, l_s in zip(self.predictions, self.golds, self.sents):
            predictions_, golds_, sents_ = [], [], []

            for e_p, e_g, e_s in zip(l_p, l_g, l_s):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)
                    sents_.append(e_s)

            self.predictions__.append(predictions_)
            self.golds__.append(golds_)
            self.sents__.append(sents_)
        # ####################

        return self.predictions__, self.golds__, self.sents__

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via test_loader.

        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        preds, golds, sents = self.predict(test_loader)
        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [int(i) for l in golds for i in l]

        print(len(sents))
        print(f'preds')
        print(len(preds))
        print(len(preds[0]))
        print(len(preds[1]))
        print('golds')
        print(len(golds))
        print(len(golds[0]))
        print(len(golds[1]))

        analysis = get_analysis(
            sents=sents,
            y_pred=preds,
            y_test=golds
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    # changed from val_losses to losses
    # but can be binary_f1 or propor_f1
    def _early_stop(self, epoch_idx, patience):
        if epoch_idx < patience:
            return False

        start = epoch_idx - patience

        # up to this index
        for count, loss in enumerate(
                self.losses[start + 1: epoch_idx + 1]):
            if loss > self.losses[start]:
                if count + 1 == patience:
                    self.early_stop_epoch = start
                    return True
            else:
                break

        return False


