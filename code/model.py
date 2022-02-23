from tqdm import tqdm
import logging

## ML specific
from torch.nn.utils.rnn import pad_packed_sequence
import torch

from transformers import BertForTokenClassification
from transformers import BertModel  # TODO next step, Bert as head

## Local imports
from utils import decode_batch
from utils import score
from utils import ez_score


class BertSimple(torch.nn.Module):
    """
    First model built for fgsa on norec_fine. 
    Expects one-hot encoded dataset, classifying num_labels outputs, w/ BertForTokenClassification
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
        label_importance = 2,       # TODO remove or tune
        output_dim=5,  # target, holder, expression, polarity, intensity
        tokenizer=None,
        targets_only = False
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
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.targets_only = targets_only

        # initialize contextual embeddings
        self.bert = BertForTokenClassification.from_pretrained(
            bert_path, num_labels=self.num_labels
        )
        self.bert.requires_grad = self.finetune
        self.bert_dropout = torch.nn.Dropout(self.dropout)

        # ensure everything is on specified device
        self.bert = self.bert.to(self.device)
        self.bert_dropout = self.bert_dropout.to(self.device)  # TODO is this needed?

        # loss function
        w = 1. + label_importance*(self.num_labels - 1)
        weight = [1/w] + [
            label_importance/w for _ in range(self.num_labels-1)
        ]  # want labels to be 2 as important as 0s
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_id,
            weight=torch.Tensor(weight),
            )

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


    def fit(self, train_loader, dev_loader=None, epochs=10):
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

                # apply loss
                logits = outputs.logits.permute(0, 2, 1)
                loss = self.backward(logits, targets)

                if b%13==0:
                    logging.info("Epoch:{:3} Batch:{:3} Loss:{}".format(epoch, b, loss.item()))
        
            if dev_loader is not None:
                self.evaluate(dev_loader)

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

        return computed_loss


    def evaluate(self, loader, verbose=False):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via loader.

        :param loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        easy_total_over_batches = 0
        hard_total_over_batches = 0


        for b, batch in enumerate(loader):
            predictions = self.predict(batch)

            # decode predictions and batch[2]
            predict_decoded = decode_batch(predictions, mask=batch[1], targets_only=self.targets_only)
            true_decoded = decode_batch(batch[2], mask=batch[1], targets_only=self.targets_only)

            annotations = ['expression', 'holder', 'polarity', 'target']

            f_target, acc_polarity, f_polarity, f_absa = score(
                true_aspect = true_decoded["targets"], 
                predict_aspect = predict_decoded["targets"], 
                true_sentiment = true_decoded["polarities"], 
                predict_sentiment = predict_decoded["polarities"], 
                train_op = False
            )

            ez = ez_score(batch[2], predictions, num_labels=self.num_labels)
            logging.debug("label f1: {}".format(ez))

            easy_total_over_batches += ez
            hard_total_over_batches += f_absa

            if not self.targets_only:
                    
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

            if verbose:
                logging.info("f_target: {}".format(f_target))
                logging.info("f_expression: {}".format(f_expression)) if not self.targets_only else None
                logging.info("f_holder: {}".format(f_holder)) if not self.targets_only else None
                logging.info("acc_polarity: {}".format(acc_polarity))
                logging.info("f_polarity: {}".format(f_polarity))

        easy_overall = easy_total_over_batches/len(loader)
        hard_overall = hard_total_over_batches/len(loader)

        logging.info("Easy overall: {easy}".format(easy=easy_overall))
        logging.info("Hard overall: {hard}".format(hard=hard_overall))

        return easy_overall, hard_overall


    def predict(self, batch):
        """
        :param batch: tensor containing batch of dev/test data 
        """
        self.eval()

        outputs = self.forward(batch)

        self.predictions = outputs.logits.argmax(2)

        return self.predictions
        

    def check_weights(self):
        weights = None
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
                                                            # logging.info("Name:{}  Module:{}  Weight:{}".format(n, m, m.weight))
                                                            weights = m.weight
                                                            break
                                                        break
                                                break
                                        break
                                break
                        break
        return weights


class BertHead(torch.nn.Module):
    """
    Abstract class that uses a BertHead w/ linear output. 
    Basically BertSimple, but able to be easily changed for more complex downstream 
    multitasking. 
    
    """
    def __init__(
        self, 
        bert_finetune=True,         # TODO tune
        bert_lr=1e-6,               # TODO tune
        bert_path="ltgoslo/norbert",  
        device="cpu",
        dropout=0.1,                # TODO tune
        ignore_id=-1,
        lr=1e-6,                    # TODO tune
        lr_scheduler_factor=0.1,    # TODO tune
        lr_scheduler_patience=2,    # TODO tune
        subtasks = ["expression", "holder", "polarity", "target"], 
        tokenizer=None,             # need tokenizer used in preprocessing 
        
        # can be **kwargs
        **kwargs
        # bidirectional=False,        # TODO tune
        # num_layers=2,               # TODO tune
    ):
        """
        Set up model specific architectures. 

        """
        super().__init__()

        self.device = device
        self.dropout = dropout  # TODO potentially refactor name?
        self.finetune = bert_finetune
        self.learning_rate = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.subtasks = subtasks
        self.tokenizer = tokenizer

        # kwargs
        self.params = self.__dict__
        self.store_kwargs(kwargs)

        # initialize bert head
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert.requires_grad = self.finetune
        self.bert_dropout = torch.nn.Dropout(self.dropout)
        
        # ensure everything is on specified device
        self.bert = self.bert.to(self.device)
        self.bert_dropout = self.bert_dropout.to(self.device)  # TODO is this needed?

        # architecture specific components
        self.components = self.init_components(self.subtasks)  # returns dict of task-specific output layers

        # loss function
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_id)

        # optimizers
        self.optimizers, self.schedulers = self.init_optimizer()  # creates same number of optimizers as output layers


    def store_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    def init_optimizer(self):
        """
        Changes with task specific architectures to optimize uniquely per subtask.
        """
        optimizers = {  # TODO create bert_lr
            "bert": torch.optim.Adam(self.bert.parameters(), lr=self.learning_rate)
        }
        schedulers = {
            "bert": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizers["bert"],
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience
            )
        }

        
        for i, (name, component) in enumerate(self.components.items()):
            opt = torch.optim.Adam(
                    component.parameters(),
                    lr=self.learning_rate   # TODO task specific learning rates?
                )                           # TODO test other optimizers?
            optimizers[name] = opt

            # learning rate scheduler to mitigate overfitting
            schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience
            ) # TODO turn on scheduler

        return optimizers, schedulers


    def fit(self, train_loader, dev_loader=None, epochs=10):
        for epoch in range(epochs):
            self.train()

            for b, batch in enumerate(train_loader):
                self.train()        # turn off eval mode
                self.zero_grad()    # clear updates from prev epoch

                predictions = self.forward(batch)
                
                # apply loss
                loss = self.backward(predictions, batch)

                # log loss every 13th batch
                if b%13==0:
                    logging.info("Epoch:{:3} Batch:{:3} Loss:{}".format(epoch, b, loss.item()))
        
            if dev_loader is not None:
                self.evaluate(dev_loader)

        logging.info("Fit complete.")


    def evaluate(self, loader, verbose=False):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via loader.

        :param loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        easy_total_over_batches = 0
        hard_total_over_batches = 0


        for b, batch in enumerate(loader):
            predictions = self.predict(batch)

            # f_target, acc_polarity, f_polarity, f_absa = score(
            #     true_aspect = true_decoded["targets"], 
            #     predict_aspect = predict_decoded["targets"], 
            #     true_sentiment = true_decoded["polarities"], 
            #     predict_sentiment = predict_decoded["polarities"], 
            #     train_op = False
            # )

            easy_scores = []

            # batch[2].shape == [32, 4, seq]

            logging.info("in eval: batch[2].shape = {}". format(batch[2].shape))
            print("in eval: batch[2].shape = {}". format(batch[2].shape))
            
            quit()

            for i, prediction in enumerate(predictions):  # [exp, hold, pol, target]
                ez = ez_score(batch[2][i], prediction, num_labels=3)
                logging.debug("label f1: {}".format(ez))

            easy_total_over_batches += ez
            hard_total_over_batches += f_absa

            if not self.targets_only:
                    
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

            if verbose:
                logging.info("f_target: {}".format(f_target))
                logging.info("f_expression: {}".format(f_expression)) if not self.targets_only else None
                logging.info("f_holder: {}".format(f_holder)) if not self.targets_only else None
                logging.info("acc_polarity: {}".format(acc_polarity))
                logging.info("f_polarity: {}".format(f_polarity))

        easy_overall = easy_total_over_batches/len(loader)
        hard_overall = hard_total_over_batches/len(loader)

        logging.info("Easy overall: {easy}".format(easy=easy_overall))
        logging.info("Hard overall: {hard}".format(hard=hard_overall))

        return easy_overall, hard_overall


    def predict(self, batch):
        """
        :param batch: tensor containing batch of dev/test data 
        """
        self.eval()

        logits = self.forward(batch)

        self.predictions = [
            output.logits.argmax(2)
            for output in logits
        ]

        return self.predictions
        

    def check_weights(self):
        """
        Helper method used for testing to check that weights get updated after loss step
        """
        single_weight = None
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
                                                            # logging.info("Name:{}  Module:{}  Weight:{}".format(n, m, m.weight))
                                                            single_weight = m.weight
                                                            break
                                                        break
                                                break
                                        break
                                break
                        break
        return single_weight


    ########### Architecture specific methods ###########
    def init_components(self, subtasks):
        """
        NOTE: New classes that overwrite this must add needed params to class doc-string.

        Parameters:
            subtasks (list(str)): subtasks the model with train for
        
        Returns:
            components (dict): output layers used for the model indexed by task name
        """

        components = {
            task: torch.nn.Linear(
                in_features=786,
                out_features=3,  # 3 possible classifications for each task
            )
            for task in subtasks
        }

        return components


    def forward(self, batch):
        """
        One forward step of training for our model.

        Parameters:
            x: token ids for a batch
        """
        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))

        hidden_states = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

        # task-specific forwards
        output = [
            self.components[task](hidden_states).permute(0, 2, 1)  # TODO double check permutation correct
            for task in self.subtasks
        ]

        logging.info("In forward, check output shapes of {num_tasks} {output_type}s".format(
            num_tasks=len(self.subtasks),
            output_type=type(output[0]))
        )
        for i in range(len(self.subtasks)):
            logging.info("shape {idx}: {shape}".format(idx=i, shape=output[0].shape))


        return output


    def backward(self, predictions, batch):
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

        true_expression = batch[2]
        true_holder = batch[3]
        true_polarity = batch[4]
        true_target = batch[5]

        logging.info("In backward")
        logging.info("predictions.shape: {}".format(predictions.shape))
        logging.info("true_expression.shape: {}".format(true_expression.shape))
        logging.info("true_holder.shape: {}".format(true_holder.shape))
        logging.info("true_polarity.shape: {}".format(true_polarity.shape))
        logging.info("true_target.shape: {}".format(true_target.shape))
        quit()

        # something probably needs to be permuted here

        self.losses = [
            self.loss(
                input=out.to(torch.device(self.device)),
                target=gold.to(torch.device(self.device))
            )
            for out, gold in zip(logits, golden)
        ]

        computed_loss = self.loss(
            input=logits.to(torch.device(self.device)),  # TODO is this type casting redundant?
            target=golden.to(torch.device(self.device))  
            )

        # calculating gradients
        computed_loss.backward()

        # updating weights from the model by calling optimizer.step()
        
        # NOTE bert optimizer alone, so needs to be updated for each task 
        # in addition to task specific optimizers
        self.optimizer.step()

        return computed_loss


class BertLSTM(torch.nn.Module):
    """
    Abstract class that uses a BertHead w/ linear output. 
    Basically BertSimple, but able to be easily changed for more complex downstream 
    multitasking. 
    
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
        num_layers=2,               # TODO tune
        bidirectional=False,        # TODO tune
        output_dim=5,  # target, holder, expression, polarity, intensity
        tokenizer=None,
    ):
        """
        Set up model specific architectures. 

        """
        super(BertSimple, self).__init__()

        self.bidirectional = bidirectional
        self.device = device
        self.dropout = bert_dropout  # TODO potentially refactor name?
        self.finetune = bert_finetune
        self.learning_rate = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.tokenizer = tokenizer

        # initialize contextual embeddings
        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = torch.nn.LSTM(
            input_size=786,  # output from BertModel is 786
            hidden_size=786, # internal hidden states can be same as input for now
            num_layers = self.num_layers,
            dropout=self.dropout,  # same dropout as Bert for simplicity
            bidirectional=bidirectional,
        )
        self.output = torch.nn.Linear(
            in_features= 786*2 if self.bidirectional else 786,
            out_features=num_labels 
        )
        self.bert.requires_grad = self.finetune
        self.bert_dropout = torch.nn.Dropout(self.dropout)

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


    def fit(self, train_loader, dev_loader=None, epochs=10):
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
                
                # apply loss
                loss = self.backward(outputs.logits.permute(0, 2, 1), targets)
                if b%13==0:
                    logging.info("Epoch:{:3} Batch:{:3} Loss:{}".format(epoch, b, loss.item()))
        
            if dev_loader is not None:
                self.evaluate(dev_loader)

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

        return computed_loss


    def evaluate(self, loader, verbose=False):
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
            predict_decoded = decode_batch(predictions, mask=batch[1], targets_only=self.targets_only)
            true_decoded = decode_batch(batch[2], mask=batch[1], targets_only=self.targets_only)

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

            if verbose:
                logging.info("f_target: {}".format(f_target))
                logging.info("f_expression: {}".format(f_expression))
                logging.info("f_holder: {}".format(f_holder))
                logging.info("acc_polarity: {}".format(acc_polarity))
                logging.info("f_polarity: {}".format(f_polarity))

            f_absa_overall = (f_absa + f_absa_overall)/2.
            logging.info("dev batch: {:3}   overall f1 absa: {}".format(b, f_absa_overall))


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
        weights = None
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
                                                            # logging.info("Name:{}  Module:{}  Weight:{}".format(n, m, m.weight))
                                                            weights = m.weight
                                                            break
                                                        break
                                                break
                                        break
                                break
                        break
        return weights

