from tqdm import tqdm
import logging

## ML specific
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
import torch

from transformers import BertForTokenClassification
from transformers import BertModel  # TODO next step, Bert as head

## Local imports
from functools import partial
from loss import DiceLoss
from utils import binary_f1
from utils import proportional_f1
from utils import score
from utils import span_f1
from utils import weighted_macro

import config



class BertSimple(torch.nn.Module):
    """
    Our first model built for fgsa on norec_fine. 
    A very simplified multitask system, w/ shared Bert embeddings

    Expects joint-labeled dataset ([B-Target, O, B-Expression, I-Expression]) 
    Classifies num_labels outputs via BertForTokenClassification
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
    Base flexible multitask learner model developed for this experiment.
    Uses a Bert head w/ linear output. 
    Basically BertSimple, except labels are expected split into subtasks.
    Easily changeable for more complex downstream subtasks
    
    Prediction outputs dictionary of subtask label predictions.
    """
    def __init__(
        self, 
        bert_finetune=True,         # TODO tune
        bert_path=config.BERT_PATH,  
        device="cuda" if torch.cuda.is_available() else "cpu",
        dropout=0.1,                # TODO tune
        ignore_id=-1,
        loss_function="cross-entropy",  # cross-entropy, dice, mse, or iou 
        lr=1e-7,                    # TODO tune
        lr_scheduler_factor=0.1,    # TODO tune
        lr_scheduler_patience=2,    # TODO tune
        subtasks = ["expression", "holder", "polarity", "target"], 
        tokenizer=None,             # need tokenizer used in preprocessing 
        
        # rest of args
        **kwargs
    ):
        """
        Set up model specific architectures. 

        """
        super().__init__()

        self.current_epoch = 0
        self.device = device
        self.dropout = dropout  # TODO potentially refactor name?
        self.finetune = bert_finetune
        self.bert_target = True if self.finetune else False
        self.learning_rate = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.subtasks = subtasks
        self.tokenizer = tokenizer
        self.ignore_id = ignore_id

        # kwargs
        self.params = self.__dict__
        self.store_kwargs(kwargs)

        # init after kwargs stored
        self.unpack_lrs()

        # initialize bert head
        self.bert = BertModel.from_pretrained(bert_path).to(torch.device(self.device))
        self.bert.requires_grad = self.finetune
        self.bert_dropout = torch.nn.Dropout(self.dropout).to(torch.device(self.device)) 
        
        # architecture specific components
        self.components = torch.nn.ModuleDict(self.init_components(self.subtasks))  # returns dict of task-specific output layers

        # loss function
        self.loss = self.get_loss(loss_function).to(torch.device(self.device))

        # optimizers
        self.optimizers, self.schedulers = self.init_optimizer()  # creates same number of optimizers as output layers

        # sanity check that subtasks only contain 4 expected
        for task in self.subtasks:
            if task in ["expression", "holder", "polarity", "target"]:
                continue
            else:
                print(f"Oh no! There is an unexpected subtask: {task}! This messes up metrics")
                raise KeyError

        # log model 
        logging.info("Subtasks: {}".format(self.subtasks))
        logging.info("Components: {}".format(self.components))
        logging.debug("Optimizers: {}".format(self.optimizers))
        logging.info("Loss: {}".format(self.loss))

    def store_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def find(self, arg, default=None):
        value = self.__dict__.get(arg)
        return value if value is not None else default

    def get_loss(self, loss_function):
        """
        Parameters:
            loss_function (str): must be either cross-entropy, mse, or dice. Otherwise returns None

        Returns:
            loss (torch.SomeLoss or None): either CrossEntropyLoss, MSELoss, or home-made IoULoss
        """
        loss = None
        label_importance = self.find("label_importance", default=None)
        weight = self.find("loss_weight", default=label_importance)

        if weight is not None:
            if isinstance(weight, list):
                weight = torch.tensor(weight).float()
            elif isinstance(weight, int):
                weight = torch.tensor([1., weight, weight])

        if loss_function is None:
            loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_id)

        elif "cross" in loss_function.lower() and weight is not None:
            loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_id, weight=weight)

        elif "cross" in loss_function.lower():
            loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_id)
        
        elif "dice" in loss_function.lower():
            loss = DiceLoss(normalization="softmax")
        
        elif "mse" in loss_function.lower():
            loss = torch.nn.MSELoss()
        
        elif "iou" in loss_function.lower():
            raise NotImplementedError()
        
        return loss

    def get_optimizer(self, optimizer_name):
        """
        Parameters:
            optimizer_name (str): Adam, AdamW, ... TODO more?

        Returns:
            optimizer (torch.optim.SomeOpt or None): the selected optimizer
        """
        optimizer = None

        if optimizer_name is None:
            params = {
                "amsgrad": self.find("amsgrad", default=False),
                "betas": self.find("betas", default=(0.9, 0.999)),
                "eps": self.find("eps", default=1e-10),
                "weight_decay": self.find("weight_decay", default=0),
            }
            optimizer = partial(torch.optim.Adam, **params)

        elif "adamw" in optimizer_name.lower():
            params = {
                "amsgrad": self.find("amsgrad", default=False),
                "betas": self.find("betas", default=(0.9, 0.999)),
                "eps": self.find("eps", default=1e-10),
                "weight_decay": self.find("weight_decay", default=0),
            }
            optimizer = partial(torch.optim.AdamW, **params)

        elif "adam" in optimizer_name.lower():
            params = {
                "amsgrad": self.find("amsgrad", default=False),
                "betas": self.find("betas", default=(0.9, 0.999)),
                "eps": self.find("eps", default=1e-10),
                "weight_decay": self.find("weight_decay", default=0),
            }
            optimizer = partial(torch.optim.Adam, **params)

        elif "sgd" in optimizer_name.lower():
            params = {
                "dampening": self.find("dampening", default=0),
                "eps": self.find("eps", default=1e-10),
                "momentum": self.find("momentum", default=0),
                "nesterov": self.find("nesterov", default=True),  # NOTE: different than torch defaults
                "weight_decay": self.find("weight_decay", default=0),
            }
            optimizer = torch.optim.SGD 

        elif "adadelta" in optimizer_name.lower():
            rho = self.find("rho", default=0.9)
            eps = self.find("eps", default=1e-10)
            optimizer = partial(torch.optim.Adadelta, rho=rho, eps=eps)
        
        return optimizer

    def unpack_lrs(self):
        if self.find("lrs") is not None:
            for task in self.lrs:
                setattr(self, task+"_lr", self.lrs[task])
            return True
        return False

    ########### Model training ###########
    def fit(self, train_loader, dev_loader=None, epochs=10):
        writer = SummaryWriter()

        for epoch in range(epochs):
            self.current_epoch = epoch

            for b, batch in enumerate(train_loader):
                self.train()        # turn off eval mode

                # feed batch to model
                output = self.forward(batch)
                
                # apply loss
                loss = self.backward(output, batch)

                # show results for first batch of each epoch
                if b==0:
                    logging.info("Epoch:{:3} Batch:{:3}".format(epoch, b))
                    for task in self.subtasks:
                        logging.info("{:10} loss:{}".format(task, loss[task].item()))
                        writer.add_scalar("loss/{}".format(task), loss[task].item(), epoch)

                    # only needed when scope loss is calculated (IMN++)
                    if self.find("scope_loss_value", default=None) is not None:
                        logging.info("{:10} loss:{}".format("scope", self.scope_loss_value.item()))
                        writer.add_scalar("loss/{}".format("scope"), self.scope_loss_value.item(), epoch)

                    if dev_loader is not None:
                        self.evaluate(dev_loader)

        logging.info("Fit complete.")
        writer.close()

    def backward(self, output, batch):
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

        true = {
            "expression": batch[2].to(torch.device(self.device)), 
            "holder": batch[3].to(torch.device(self.device)),
            "polarity": batch[4].to(torch.device(self.device)),
            "target": batch[5].to(torch.device(self.device)),
        }

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        for task in self.optimizers:
            self.optimizers[task].zero_grad()

        # calcaulate losses per task
        self.losses = {
            task: self.loss(
                input=output[task],
                target=true[task]
            )
            for task in self.subtasks
        }
        loss_total = torch.zeros(1).to(torch.device(self.device))
        for task in self.subtasks:
            loss_total += self.loss(
                input=output[task],
                target=true[task]
            )
        self.losses["total"] = loss_total


        # calculate gradients for parameters used per task
        for task in self.subtasks:
            self.losses[task].backward(retain_graph=True)  # retain_graph needed to update shared tasks
        # self.loss_total.backward()

        # TODO should there be bert optimizer alone, 
        # if so needs to be updated for each task 
        # in addition to task specific optimizers

        # update weights from the model by calling optimizer.step()
        for task in self.optimizers:
            self.optimizers[task].step()

        return self.losses

    def evaluate(self, loader, verbose=False):
        """
        Returns overall binary and proportional F1 scores for predictions on the
        development data via loader, while logging task-wise scores along the way.

        F1 variants:
            ABSA: overall F1 score as measured by IMN and RACL
            Binary: binary over F1 looking only for an overlap of correctly estimated labels
            Hard: averaged task wise F1 as measured by IMN and RACL
            Macro: averaged task wise f1_score from sklearn using param average='weights'
            Proportional: token wise f1_score from sklearn using param average='micro'
            Span: 'nicer' version of scope checking similar to RACL and IMN metrics TODO rewrite

        Parameters:
            loader (torch.DataLoader): 

        Return:
            absa_overall, binary_overall, hard_overall, macro_overall, proportional_overall, span_overall
        """
        absa_total_over_batches = 0  # f_absa from score() used in RACL experiment across batches
        binary_total_over_batches = 0  # avg of easy task-wise f1-scores across batches
        macro_total_over_batches = 0  # avg of easy task-wise f1-scores across batches
        proportional_total_over_batches = 0  # avg of easy task-wise f1-scores across batches
        span_total_over_batches = 0  # avg of easy task-wise f1-scores across batches
        hard_total_over_batches = 0  # avg of hard task-wise f1-scores across batches


        for b, batch in enumerate(loader):
            preds, golds = self.predict(batch)

            ### hard score
            hard = {}
            if "target" in self.subtasks and "polarity" in self.subtasks:
                f_target, acc_polarity, f_polarity, f_absa = score(
                    true_aspect = golds["target"], 
                    predict_aspect = preds["target"], 
                    true_sentiment = golds["polarity"], 
                    predict_sentiment = preds["polarity"], 
                    train_op = False,
                    no_neutral = True  # NOTE change me when using English data w/ neutral polarities 
                ) 

                hard["target"] = f_target
                hard["polarity"] = f_polarity

                logging.debug("{:10} hard: {}".format("target", f_target))
                logging.debug("{:10} hard: {}".format("polarity", f_polarity))
                logging.debug("{:10} acc : {}".format("polarity", acc_polarity))
            else:
                f_absa = 0

            if "expression" in self.subtasks:
                f_expression, _, _, _ = score(
                    true_aspect = golds["expression"], 
                    predict_aspect = preds["expression"], 
                    true_sentiment = golds["polarity"], 
                    predict_sentiment = preds["polarity"], 
                    train_op = True,
                    no_neutral = True  # NOTE change me when using English data w/ neutral polarities 
                )

                hard["expression"] = f_expression

                logging.debug("{:10} hard: {}".format("expression", f_expression))

            if "holder" in self.subtasks:
                f_holder, _, _, _ = score(
                    true_aspect = golds["holder"], 
                    predict_aspect = preds["holder"], 
                    true_sentiment = golds["polarity"], 
                    predict_sentiment = preds["polarity"], 
                    train_op = True,
                    no_neutral = True  # NOTE change me when using English data w/ neutral polarities 
                )

                hard["holder"] = f_holder

                logging.debug("{:10} hard: {}".format("holder", f_holder))

            ### binary overlap, proportional f1, span f1, and weighted macro f1 (sklearn)
            binary = {}
            proportional = {}
            span = {}
            macro = {}
            easy = {}
            for task in self.subtasks: 
                b = binary_f1(golds[task], preds[task])
                p = proportional_f1(golds[task], preds[task], num_labels=3)
                s = span_f1(golds[task], preds[task])
                m = weighted_macro(golds[task], preds[task], num_labels=3)
                binary[task] = b
                proportional[task] = p
                span[task] = s
                macro[task] = m
                logging.debug("{task:10}       binary: {score}".format(task=task, score=b))
                logging.debug("{task:10} proportional: {score}".format(task=task, score=p))
                logging.debug("{task:10}         span: {score}".format(task=task, score=s))
                logging.debug("{task:10}        macro: {score}".format(task=task, score=m))

            # NOTE add average f1 over score per subtask to totals
            absa_total_over_batches += f_absa
            binary_total_over_batches += (sum([binary[task] for task in self.subtasks])/len(self.subtasks))
            proportional_total_over_batches += (sum([proportional[task] for task in self.subtasks])/len(self.subtasks))
            span_total_over_batches += (sum([span[task] for task in self.subtasks])/len(self.subtasks))
            macro_total_over_batches += (sum([macro[task] for task in self.subtasks])/len(self.subtasks))
            hard_total_over_batches += (sum([hard[task] for task in self.subtasks])/len(self.subtasks))

        absa_overall = absa_total_over_batches/len(loader)
        binary_overall = binary_total_over_batches/len(loader)
        proportional_overall = proportional_total_over_batches/len(loader)
        span_overall = span_total_over_batches/len(loader)
        macro_overall = macro_total_over_batches/len(loader)
        hard_overall = hard_total_over_batches/len(loader)

        logging.info(" (RACL) ABSA overall: {absa}".format(absa=absa_overall))
        logging.info(" (RACL) Hard overall: {hard}".format(hard=hard_overall))

        logging.info("      Binary overall: {binary}".format(binary=binary_overall))
        logging.info("Proportional overall: {proportional}".format(proportional=proportional_overall))
        logging.info("        Span overall: {span}".format(span=span_overall))
        logging.info("       Macro overall: {macro}".format(macro=macro_overall))
        
        print(" (RACL) ABSA overall: {absa}".format(absa=absa_overall))
        print(" (RACL) Hard overall: {hard}".format(hard=hard_overall))

        print("      Binary overall: {binary}".format(binary=binary_overall))
        print("Proportional overall: {proportional}".format(proportional=proportional_overall))
        print("        Span overall: {span}".format(span=span_overall))
        print("       Macro overall: {macro}".format(macro=macro_overall))
        

        return absa_overall, binary_overall, hard_overall, macro_overall, proportional_overall, span_overall

    def predict(self, batch):
        """
        Parameters:
            batch: tuple of batch tensors (size 6) w/ dev/test data 
        """
        self.eval()

        outputs = self.forward(batch)

        prediction_tensors = {
            task: outputs[task].argmax(1)
            for task in self.subtasks
        }

        self.golds = {task: [] for task in self.subtasks}
        self.preds = {task: [] for task in self.subtasks}

        true = {
            "expression": batch[2], 
            "holder": batch[3],
            "polarity": batch[4],
            "target": batch[5],
        }


        # strip away padding
        for i, row in enumerate(batch[0]):
            for t, token in enumerate(row):
                if token.item() == 0:  # padding id is 0
                    for task in self.subtasks:
                        self.preds[task].append(
                            prediction_tensors[task][i][:t].tolist()
                        )
                        self.golds[task].append(
                            true[task][i][:t].tolist()
                        )
                    break

        return self.preds, self.golds
        
    def score(self, X, y):
        """
        NOTE Made for GridSearch, but never used
        TODO Delete? or Use?
        """
        absa, easy, hard = self.evaluate(X)
        s = absa if y == "absa" else None
        s = easy if y == "easy" else None
        s = hard if y == "hard" else None
        return s

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
        Parameters:
            subtasks (list(str)): subtasks the model with train for
        
        Returns:
            components (dict): output layers used for the model indexed by task name
        """

        components = torch.nn.ModuleDict({
            task: torch.nn.ModuleDict({
                "linear": torch.nn.Linear(
                    in_features=768,
                    out_features=3,  # 3 possible classifications for each task
                ).to(torch.device(self.device))
            })
            for task in subtasks
        })

        return components

    def init_optimizer(self):
        """
        Changes with task specific architectures to optimize uniquely per subtask.
        """
        other_components = self.find("other_components", default={
            component: {"lr": self.learning_rate, "tasks": []}
            if component not in self.subtasks else {}
            for component in self.components
        })
        optimizer_name = self.find("optimizer_name", default=self.find("optimizer"))
        bert_tuners = {  # subtasks to train bert on
            task: self.find(f"bert_{task}", default=False)
            for task in self.subtasks
        }
        task_lrs = { # subtask specific learning rates
            task: self.find(task+"_learning_rate", default=self.find(task+"_lr")) 
            for task in self.subtasks
        }

        optimizers = {}
        schedulers = {}

        # single optimizer type for all subtasks for simplicity
        optimizer = self.get_optimizer(optimizer_name)
        
        for task in self.subtasks:
            lr = task_lrs[task] if task_lrs.get(task) is not None else self.learning_rate

            optimizers[task] = optimizer(
                self.components[task].parameters(),
                lr=lr 
            )

            if bert_tuners[task] and self.finetune:
                optimizers[task].add_param_group(
                    # use main learning rate for bert training
                    {"params": self.bert.parameters(), "lr":self.learning_rate}
                )

            # learning rate scheduler to mitigate overfitting
            schedulers[task] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizers[task],
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True,
                eps=1e-10,
            )

        # make sure non-single-task-specific components are added to optimizers
        if other_components is not None:
            for component in other_components:
                if other_components[component] is not {}:
                    lr = other_components[component].get("lr")
                    lr = lr if lr is not None else self.learning_rate

                    # IMN architecture
                    if component == "shared":
                        tasks = other_components[component].get("tasks")
                        tasks = tasks if tasks is not None else []

                        for task in tasks:
                            optimizers[task].add_param_group(
                                {"params": self.components[component].parameters(), "lr":lr}
                            )
                   
                    # RACL architecture
                    elif component == "relations":
                        for stack in self.components["relations"]:
                            for layer in self.components["relations"][stack]:
                                for task in self.subtasks:
                                    if task in layer:
                                        print("adding {} to optimizer {}".format(layer, task))
                                        optimizers[task].add_param_group(
                                            {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                        )

                                        # only passes this if when layer=shared_at_polarity
                                        if "shared" in layer:
                                            print("adding {} to optimizer {}".format(layer, task))

                                            optimizers["target"].add_param_group(
                                                {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                            )
                                            optimizers["expression"].add_param_group(
                                                {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                            )


        return optimizers, schedulers

    def forward(self, batch):
        """
        One forward step of training for our model.

        Parameters:
            batch: entire batch object for cleaner self.fit()
        """
        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))

        embeddings = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        embeddings = self.bert_dropout(embeddings)

        # task-specific forwards
        output = {}
        for task in self.subtasks:
            # iterate over layers of task-specific components
            hidden_task_states = embeddings
            for name in self.components[task]:
                layer = self.components[task][name]
                hidden_task_states = layer(hidden_task_states).permute(0, 2, 1)
            output[task] = hidden_task_states

        return output


class FgsaLSTM(BertHead):
    """
    Simple iteration on BertHead, replacing linear output w/ LSTMs.
    Increase complexity as simply as possible (testing flexability).

    Prediction outputs dictionary of subtask label predictions.
    """

    def init_components(self, subtasks):
        """
        Parameters:
            subtasks (list(str)): subtasks the model with train for
        
        Returns:
            components (dict): output layers used for the model indexed by task name
        """
        bidirectional = self.find("bidirectional", default=False)
        hidden_size = self.find("hidden_size", default=768)
        num_layers = self.find("num_layers", default=3)
        dropout = self.find("dropout", default=0.1)

        components = torch.nn.ModuleDict({
            task: torch.nn.ModuleDict({
                # cannot use nn.Sequential since LSTM outputs a tuple of last hidden layer and final cell states
                "lstm": torch.nn.LSTM(
                    input_size=768,
                    hidden_size=hidden_size,  # Following BERT paper
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional, 
                ).to(torch.device(self.device)),
                "linear": torch.nn.Linear(
                    in_features=hidden_size*2 if bidirectional else hidden_size,
                    out_features=3,
                ).to(torch.device(self.device))
            })
            for task in subtasks
        })

        return components
    
    def forward(self, batch):
        """
        One forward step of training for our model.
        NOTE: torch.nn.LSTMs output a tuple, where only the first element is needed for classification

        Parameters:
            batch: entire batch object for cleaner self.fit()
        """
        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))

        embeddings = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        embeddings = self.bert_dropout(embeddings)

        # task-specific forwards
        output = {}
        for task in self.subtasks:
            hidden, _ = self.components[task]["lstm"](embeddings)
            output[task] = self.components[task]["linear"](hidden).permute(0, 2, 1)

        return output


class IMN(BertHead):
    """
    Similar to the original IMN structure, just built on a BertHead (in PyTorch).

    Parameters:
        interactions (int, default=2)
        expression_layers (int, default=2)
        polarity_layers (int, default=2)
        shared_layers (int, default=2)
        target_layers (int, default=2)
    """
    def __init__(self, subtasks = ["target", "expression", "polarity"], **kwargs):
        # overwrite BertHead class default for subtasks
        kwargs["subtasks"] = subtasks
        super(IMN, self).__init__(**kwargs)

    def init_components(self, subtasks):
        """
        Every component 
        Parameters:
            subtasks (list(str)): subtasks the model with train for
        
        Returns:
            components (dict): output layers used for the model indexed by task name
        """
        cnn_dim = self.find("cnn_dim", default=768)
        expression_layers = self.find("expression_layers", default=2)
        polarity_labels = self.find("polarity_labels", default=3)  # expands to 5 when using english data sets
        polarity_layers = self.find("polarity_layers", default=2)
        shared_layers = self.find("shared_layers", default=2)
        target_layers = self.find("target_layers", default=2)
        # scope specific
        scope_lr = self.find("scope_lr", default=self.learning_rate)
        optimizer_name = self.find("optimizer_name", default="adam")

        components = torch.nn.ModuleDict({
            "shared": torch.nn.ModuleDict({
                # shared convolutions over embeddings
            }),
            "target": torch.nn.ModuleDict({
                # aspect extraction: dropout -> cnn -> cat -> dropout -> cnn -> cat -> dropout -> linear
            }),
            "expression":torch.nn.ModuleDict({
                # opinion extraction: dropout -> cnn -> cat -> dropout -> cnn -> cat -> dropout -> linear
                # this is really done jointly w/ target, but separate here for more flexibility
            }),
            "polarity":torch.nn.ModuleDict({
                # polarity classification: cnn -> attention -> cat -> dropout -> linear
            }),
            # doc-level skipped for now
        })

        ######################################
        # Shared CNN layers
        ######################################
        for i in range(shared_layers):
            print("Shared CNN layer {}".format(i))
            layer = torch.nn.ModuleDict({
                f"dropout": torch.nn.Dropout(self.dropout).to(torch.device(self.device)),
            })
            if i == 0:
                layer[f"cnn_{i}_3"] = torch.nn.Conv1d(
                    in_channels = int(768),
                    out_channels = int(cnn_dim/2), 
                    kernel_size = 3,
                    padding=1
                ).to(torch.device(self.device))

                
                layer[f"cnn_{i}_5"] = torch.nn.Conv1d(
                    in_channels = int(768),
                    out_channels = int(cnn_dim/2), 
                    kernel_size = 5,
                    padding=2,
                ).to(torch.device(self.device))
            
            else:
                
                layer[f"cnn_{i}_5"] = torch.nn.Conv1d(
                    in_channels = cnn_dim,
                    out_channels = cnn_dim, 
                    kernel_size = 5,
                    padding=2,
                ).to(torch.device(self.device))
            components["shared"].update(layer)

        self.other_components = {
            "shared": {
                "lr": self.learning_rate,
                "tasks": self.subtasks

            }
        }

        #######################################
        # Task-specific CNN layers
        #######################################
        layers = []
        for layer in range(target_layers):
            # every layer gets a dropout, cnn, and relu activation
            layers.append(torch.nn.Dropout(self.dropout))
            layers.append(torch.nn.Conv1d(
                    in_channels = cnn_dim,
                    out_channels = cnn_dim, 
                    kernel_size = 5,
                    padding=2,
                ))
            layers.append(torch.nn.ReLU())
        target_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["target"].update({"cnn_sequential": target_sequential})

        layers = []
        for layer in range(polarity_layers):
            # every layer gets a dropout, cnn, and relu activation
            layers.append(torch.nn.Dropout(self.dropout))
            layers.append(torch.nn.Conv1d(
                    in_channels = cnn_dim,
                    out_channels = cnn_dim, 
                    kernel_size = 5,
                    padding=2,
                ))
            layers.append(torch.nn.ReLU())
        polarity_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["polarity"].update({"cnn_sequential": polarity_sequential})
        
        layers = []
        for layer in range(expression_layers):
            # every layer gets a dropout, cnn, and relu activation
            layers.append(torch.nn.Dropout(self.dropout))
            layers.append(torch.nn.Conv1d(
                    in_channels = cnn_dim,
                    out_channels = cnn_dim, 
                    kernel_size = 5,
                    padding=2,
                ))
            layers.append(torch.nn.ReLU())
        expression_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["expression"].update({"cnn_sequential": expression_sequential})

        #######################################
        # Task-specific output layers
        #######################################
        components["target"].update({
            "linear": torch.nn.Sequential(
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(
                    in_features=int(768+(shared_layers+1)*cnn_dim),  # bert:768 + shared_cnn:(300 + 300) + target_cnn:300
                    out_features=3
                ), 
                torch.nn.Softmax(dim=-1)
            ).to(torch.device(self.device))
        })

        components["expression"].update({
            "linear": torch.nn.Sequential(
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(
                    in_features=int(768+(shared_layers+1)*cnn_dim), # bert:768 + shared_cnn:(300 + 300) + expression_cnn:300
                    out_features=3
                ), 
                torch.nn.Softmax(dim=-1)
            ).to(torch.device(self.device))
        })

        # polarity had attention before linear
        components["polarity"].update({
            "attention": torch.nn.MultiheadAttention(cnn_dim, num_heads=1).to(torch.device(self.device)), 
            "linear": torch.nn.Sequential(
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(
                    in_features=int(2*cnn_dim), # initial_shared_features:300 + polarity_cnn:300
                    out_features=polarity_labels  # NOTE: SemEval data has neutral and confusing polarities
                ), 
                torch.nn.Softmax(dim=-1)
            ).to(torch.device(self.device))
        })

        
        #######################################
        # Scope predictions
        #######################################

        components.update({
            "scope": torch.nn.ModuleDict({
                # scope finder: shared -> linear 
                "linear": torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=cnn_dim,
                        out_features=2
                    ),
                    torch.nn.Sigmoid()
                ).to(torch.device(self.device))
            })
        })

        self.relu = torch.nn.ReLU()
        self.scope_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_id)
        scope_optimizer = self.get_optimizer(optimizer_name)
        self.scope_optimizer = scope_optimizer(
            components["scope"]["linear"].parameters(),
            lr=scope_lr
        )

        #######################################
        # Re-encoder
        #######################################
        components["shared"].update({
            "re_encode": torch.nn.Sequential(
                torch.nn.Linear(
                    # sentence_output:cnn_dim + target_output:3 + expression_output:3 + polarity_output:5
                    in_features=int(cnn_dim + 3 + 3 + polarity_labels),  
                    out_features=cnn_dim,
                ),
                torch.nn.ReLU()
            ).to(torch.device(self.device))
        })

        return components

    def forward(self, batch):
        cnn_dim = self.find("cnn_dim", default=768)
        interactions = self.find("interactions", default=2)
        expression_layers = self.find("expression_layers", default=2)
        polarity_layers = self.find("polarity_layers", default=2)
        shared_layers = self.find("shared_layers", default=2)
        target_layers = self.find("target_layers", default=2)
        # attention params
        queries = self.find("queries", default=batch[4])  # batch[4] = polarity

        input_ids = batch[0].to(torch.device(self.device))
        mask = batch[1].to(torch.device(self.device))

        # NOTE: System maintains sequence size by
        # expanding/re-encoding to embedding size 
        
        #########################################
        # Shared word embedding layer 
        #########################################
        try:
            word_embeddings = self.bert(
                input_ids = input_ids,
                attention_mask = mask,
            ).last_hidden_state.to(torch.device(self.device))
        except Exception as e:
            print("input_ids {}".format(input_ids.max().item()))
            raise e
        word_embeddings = self.bert_dropout(word_embeddings)

        # NOTE permute so shape is [batch, embedding, sequence] into cnn
        # then permute back to [batch, sequence, embedding] for attnetion
        word_embeddings = word_embeddings.permute(0, 2, 1)
        sentence_output = word_embeddings  # TODO detach and/or clone?

        ######################################
        # Shared CNN layers
        ######################################
        shared = self.components["shared"]
        for i in range(shared_layers):
            if i == 0:
                # dropout w/ 3 cnn and 5 cnn
                sentence_output = shared["dropout"](sentence_output)

                sentence_output_3 = shared[f"cnn_{i}_3"](sentence_output)
                sentence_output_5 = shared[f"cnn_{i}_5"](sentence_output)

                sentence_output = torch.cat((sentence_output_3, sentence_output_5), dim=1)  # cat embedding dim
            else:
                # just dropout and cnn5
                sentence_output = shared["dropout"](sentence_output)
                sentence_output = shared[f"cnn_{i}_5"](sentence_output)

            # update word embeddings with shared features learned from this cnn layer
            word_embeddings = torch.cat((word_embeddings, sentence_output), dim=1) # cat embedding dim

        # only the information learned from shared cnn(s), no embeddings
        initial_shared_features = sentence_output  # TODO detach and/or clone?    


        if self.find("find_scope", default=True):
            scope_output = self.scope_relevance(
                batch,
                sentence_output
            ) 
            # sentence_output.shape = [batch, embedding (768), sequence]
            sentence_output *= scope_output.unsqueeze(1).expand(-1, sentence_output.size(1), -1)


        #######################################
        # Task-specific layers
        #######################################
        self.output = {}  # task-specific outputs stored along the way
        softmax = torch.nn.Softmax(dim=-1)  # expecting labels to be last dim # TODO move to init_components()

        for i in range(interactions+1):
            ### Subtask: target
            target_output = sentence_output
            if target_layers > 0:
                target_cnn_output = self.components["target"]["cnn_sequential"](target_output)
                target_output = target_cnn_output

            target_output = torch.cat((word_embeddings, target_output), dim=1)  # cat embedding dim
            ### target_output.shape = [batch, sequence, embedding]
            target_output = target_output.permute(0, 2, 1)  # batch, sequence, embedding
            target_output = self.components["target"]["linear"](target_output)
            self.output["target"] = target_output.permute(0, 2, 1)  # batch, labels, sequence


            ### Subtask: expression
            expression_output = sentence_output
            if expression_layers > 0:
                expression_cnn_output = self.components["expression"]["cnn_sequential"](expression_output)
                expression_output = expression_cnn_output

            expression_output = torch.cat((word_embeddings, expression_output), dim=1)  # cat embedding dim
            expression_output = expression_output.permute(0, 2, 1)  # batch, sequence, embedding
            expression_output = self.components["expression"]["linear"](expression_output)
            self.output["expression"] = expression_output.permute(0, 2, 1)  # batch, labels, sequence


            ### Subtask: polarity
            polarity_output = sentence_output

            if polarity_layers > 0:
                polarity_cnn_output = self.components["polarity"]["cnn_sequential"](polarity_output)
                polarity_output = polarity_cnn_output

            # attention block
            # values = polarity_output.permute(2, 0, 1)
            queries, keys, values = self.get_attention_inputs(
                target_cnn_output.permute(2, 0, 1),     # sequence, batch, embedding
                expression_cnn_output.permute(2, 0, 1), 
                polarity_cnn_output.permute(2, 0, 1),
                batch
            )

            polarity_output, _ = self.components["polarity"]["attention"](
                queries,    # query, i.e. polar cnn output w/ weights
                keys,       # keys, i.e. polar cnn output for self attention
                values,     # values, i.e polar cnn output w/ probabilities for B and I tags
                need_weights=False,
                key_padding_mask=(
                    (batch[1]*-1)+1  # switch [11100] to [00011]
                ).bool().to(torch.device(self.device)),
            )
            polarity_output = polarity_output.permute(1, 2, 0)  # batch, embedding, sequence

            # NOTE: concat w/ initial_shared_features not word_embeddings like in target
            polarity_output = torch.cat((initial_shared_features, polarity_output), dim=1)  # cat embedding dim
            polarity_output = polarity_output.permute(0, 2, 1)  # batch, sequence, embedding
            polarity_output = self.components["polarity"]["linear"](polarity_output)
            self.output["polarity"] = polarity_output.permute(0, 2, 1)  # batch, labels, sequence

            # update sentence_output for next iteration
            sentence_output = torch.cat(
                (
                    sentence_output,            # batch, embedding, sequence
                    self.output['target'], 
                    self.output['expression'], 
                    self.output["polarity"],
                ),
                dim=1
            ).permute(0, 2, 1)  # batch, sequence, embedding

            # re-encode embedding dim to expected cnn dimension
            sentence_output = self.components["shared"]["re_encode"](sentence_output).permute(0, 2, 1)  # batch, embedding, sequence

        return self.output

    def get_attention_inputs(self, target, expression, polarity, batch):
        """ """
        gold_transmission = self.find("gold_transmission", default=False)

        query = self.find("query", default=self.find("queries", default="polarity"))
        key = self.find("key", default=self.find("keys", default="polarity"))
        value = self.find("value", default=self.find("values", default="polarity"))

        queries, keys, values = None, None, None 

        # Q
        if "polar" in query:
            queries = polarity
        elif "target" in query:
            queries = target
        elif "expression" in query:
            queries = expression
        else:
            queries = polarity

        # K
        if "polar" in key:
            keys = polarity
        elif "target" in key:
            keys = target
        elif "expression" in key:
            keys = expression
        else:
            keys = polarity

        # v
        if "polar" in value:
            values = polarity
            true_values = batch[4].permute(1,0).to(torch.device(self.device))
        elif "target" in value:
            values = target
            true_values = batch[5].permute(1,0).to(torch.device(self.device))
        elif "expression" in value:
            values = expression
            true_values = batch[2].permute(1,0).to(torch.device(self.device))
        else:
            values = polarity

        # BUG: size problem
        # values is of size [batch, cnn_dim, sequence]
        # true_values is of size [batch, sequence]
        if gold_transmission:
            gold_influence = self.get_prob(self.current_epoch, self.find("warm_up_constant", default=5))
            values = (  # strengthen signal for true values
                gold_influence*true_values.bool().float().unsqueeze(-1).expand(values.shape)
                 + (1-gold_influence)*values
            ).detach().to(torch.device(self.device))
            values.requires_grad = True


        return queries, keys, values

    def scope_relevance(self, batch, shared_output) -> tuple():
        """
        Return:
            scope_loss_value: value of loss for current scope prediction
            scope_logits: scope predictions after shared layers
            scope_true: true scope for current batch
        """
        gold_transmission = self.find("gold_transmission", default=False)

        labels = {
            "expression": batch[2],
            "polarity": batch[4],
            "target": batch[5],
        }

        self.scope_true = self.relu(
            (labels["expression"] + labels["polarity"] + labels["target"])
        ).bool().long().to(torch.device(self.device))
        # scope_true.shape = [batch, sequence]

        # shared_output.shape = [batch, embedding (768), sequence]
        self.scope_logits = self.components["scope"]["linear"](shared_output.permute(0, 2, 1))

        # scope_logits.shape = [batch, sequence, 1]
        self.scope_loss_value = self.scope_loss(self.scope_logits.permute(0, 2, 1), self.scope_true)
        self.scope_loss_value.backward(retain_graph=True)
        self.scope_optimizer.step()

        if gold_transmission:
            gold_influence = self.get_prob(self.current_epoch, self.find("warm_up_constant", default=5))
            self.scope_output = (
                gold_influence*self.scope_true + (1-gold_influence)*self.scope_logits.argmax(-1).squeeze(-1)
            ).detach()
            self.scope_output.requires_grad = True
        else:
            self.scope_output = self.scope_logits.argmax(-1).squeeze(-1).detach()

        return self.scope_output.to(torch.device(self.device))

    @staticmethod
    def get_prob(epoch_count, constant=5):
        """
        Borrowed directly from https://github.com/ruidan/IMN-E2E-ABSA
        
        Compute the probability of using gold opinion labels in opinion transmission
        (To alleviate the problem of unreliable predictions of opinion labels sent from AE to AS,
        in the early stage of training, we use gold labels as prediction with probability 
        that depends on the number of current epoch)

        """
        prob = constant/(constant+torch.exp(torch.tensor(epoch_count).float()/constant))
        return prob


class RACL(IMN):
    """
    Similar to original RACL architecture, just built on a BertHead (in PyTorch).

    Parameters:
        TODO fill in parameters used in self.find()
    """

    def init_components(self, subtasks):
        cnn_dim = self.find("cnn_dim", default=768)
        stack_count = self.find("stack_count", default=1)

        components = torch.nn.ModuleDict({
            "target": torch.nn.ModuleDict({
                # aspect extraction: cnn -> relu -> matmul w/ expression -> attention -> cat -> linear
                "cnn": torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels = int(768),
                        out_channels = int(cnn_dim), 
                        kernel_size = 5,
                        padding=2
                    ),
                    torch.nn.ReLU()
                ).to(torch.device(self.device)),
                "linear": torch.nn.Sequential(
                    torch.nn.Linear(
                    in_features=cnn_dim*2,  # TODO this will be doubled..?
                    out_features=3
                    ),
                    # torch.nn.Sigmoid(),  # TODO activation after linear?
                ).to(torch.device(self.device)),
                "re_encode": torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=int(cnn_dim*2), 
                        out_features=int(768)
                    ),
                    # torch.nn.ReLU(),  # TODO activate like l2 norm or nah?
                    torch.nn.AlphaDropout(
                        self.dropout,
                    ).to(torch.device(self.device)),
                ).to(torch.device(self.device))
            }),
            "expression":torch.nn.ModuleDict({
                # opinion extraction: cnn -> relu -> matmul w/ target -> attention -> cat -> linear
                "cnn": torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels = int(768),
                        out_channels = int(cnn_dim), 
                        kernel_size = 5,
                        padding=2
                    ),
                    torch.nn.ReLU()
                ).to(torch.device(self.device)),
                "linear": torch.nn.Sequential(
                    torch.nn.Linear(
                    in_features=cnn_dim*2,
                    out_features=3
                    ),
                    # torch.nn.Sigmoid(),  # TODO activation after linear?
                ).to(torch.device(self.device)),
                "re_encode": torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=int(cnn_dim*2), 
                        out_features=int(768)
                    ),
                    # torch.nn.ReLU(),  # TODO activate like l2 norm or nah?
                    torch.nn.AlphaDropout(
                        self.dropout,
                    ),
                ).to(torch.device(self.device))
            }),
            "polarity":torch.nn.ModuleDict({
                # polarity classification: cnn -> relu -> matmul w/ (embedding) -> attention -> cat -> dropout -> linear
                "cnn": torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels = int(768),
                        out_channels = int(cnn_dim), 
                        kernel_size = 5,
                        padding=2
                    ),
                    torch.nn.ReLU()
                ).to(torch.device(self.device)),
                "linear": torch.nn.Sequential(
                    torch.nn.Linear(
                    in_features=cnn_dim, # TODO double?
                    out_features=3
                    ),
                    # torch.nn.Sigmoid(),  # TODO activation after linear?
                ).to(torch.device(self.device)),
                "re_encode": torch.nn.Sequential(
                    torch.nn.Linear(  # TODO delete
                        in_features=int(cnn_dim), 
                        out_features=768
                    ),
                    # torch.nn.ReLU(),  # TODO activate like l2 norm or nah?
                    torch.nn.AlphaDropout(
                        self.dropout,
                    ).to(torch.device(self.device)),
                ).to(torch.device(self.device))
            }),
        })


        ######################################
        # Shared relation components
        ######################################
        components.update(torch.nn.ModuleDict({
            "relations": torch.nn.ModuleDict({
                f"stack_{i}": torch.nn.ModuleDict({
                    "target_at_expression": torch.nn.MultiheadAttention(
                        embed_dim = cnn_dim,
                        num_heads = 1,
                        dropout=self.dropout,
                        # query: torch.norm(expression)  # NOTE L2 norm
                        # key: torch.norm(target)
                        # values: target
                        # key_padding_mask: (batch[1]*-1)+1

                        # qk-outputs used in group v-mul later
                    ).to(torch.device(self.device)),
                    "expression_at_target": torch.nn.MultiheadAttention(
                        embed_dim = cnn_dim,
                        num_heads = 1,
                        dropout=self.dropout,
                        # query: torch.norm(target)  # NOTE L2 norm
                        # key: torch.norm(expression)
                        # values: expression
                        # key_padding_mask: (batch[1]*-1)+1
                    ).to(torch.device(self.device)),
                    "shared_at_polarity": torch.nn.MultiheadAttention(
                        embed_dim = cnn_dim,
                        num_heads = 1,
                        dropout=self.dropout,
                        # query: shared_hidden_state
                        # key: torch.norm(polarity_cnn_output)
                        # after softmax, keys joined w/ target_at_expression & (expression_probs expanded to cnn_dim)
                        # value: polarity
                        # key_padding_mask: (batch[1]*-1)+1
                    ).to(torch.device(self.device)),
                })
                for i in range(stack_count)
            }),
        }))

        self.other_components = {
            "relations": {
                "lr": self.learning_rate
            }
        }

        return components

    def forward(self, batch):
        gold_transmission = self.find("gold_transmission", default=False)
        stack_count = self.find("stack_count", default=1)

        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))

        embeddings = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        embeddings = self.bert_dropout(embeddings).permute(0, 2, 1)

        # store inputs along the way 
        expression_inputs = [embeddings]
        polarity_inputs = [embeddings]
        target_inputs = [embeddings]
        shared_query = [embeddings]

        # store outputs along the way 
        expression_outputs = []
        polarity_outputs = []
        target_outputs = []

        
        for i in range(stack_count):
            # target & expression convolution
            target_cnn = self.components["target"]["cnn"](target_inputs[-1])
            expression_cnn = self.components["expression"]["cnn"](expression_inputs[-1])

            # Relation R1  # TODO remove weights expression_at_target (not needed)
            target_attn, target_at_expression = self.components["relations"][f"stack_{i}"]["target_at_expression"](
                # expects shape: [seq, batch, cnn_dim]
                query=torch.nn.functional.normalize(expression_cnn, p=2, dim=-1).permute(2, 0, 1),
                key=torch.nn.functional.normalize(target_cnn, p=2, dim=-1).permute(2, 0, 1),
                value=target_cnn.permute(2, 0, 1),
                key_padding_mask=((batch[1]*-1)+1).bool().to(torch.device(self.device)),
                need_weights=True
            )

            target_inter = torch.cat((target_cnn.permute(0, 2, 1), target_attn.permute(1,0,2)), dim=-1)
            target_logits = self.components["target"]["linear"](target_inter)

            # TODO remove weights expression_at_target (not needed)
            expression_attn, expression_at_target = self.components["relations"][f"stack_{i}"]["expression_at_target"](
                # expects shape: [seq, batch, cnn_dim]
                query=torch.nn.functional.normalize(expression_cnn, p=2, dim=-1).permute(2, 0, 1),
                key=torch.nn.functional.normalize(target_cnn, p=2, dim=-1).permute(2, 0, 1),
                value=expression_cnn.permute(2, 0, 1),
                key_padding_mask=((batch[1]*-1)+1).bool().to(torch.device(self.device)),
                need_weights=False
            )
            expression_inter = torch.cat((expression_cnn.permute(0, 2, 1), expression_attn.permute(1,0,2)), dim=-1)
            expression_logits = self.components["expression"]["linear"](expression_inter)

            # TODO gold transmission of expressions
            if gold_transmission:
                raise NotImplementedError  # not exactly complete, come back and check
                expression_confidence = self.get_confidence(
                    expression_logits, 
                    batch,
                    self.current_epoch
                )
                expression_propagate = expression_at_target*(  # expand confidence to shape of attention weights
                    expression_confidence.unsqueeze(-1).expand(expression_at_target.shape)
                )
            else:
                expression_propagate = expression_attn

            # polarity convolution
            polarity_cnn = self.components["polarity"]["cnn"](polarity_inputs[-1])

            # shared-polarity attention
            polarity_attn, _ = self.components["relations"][f"stack_{i}"]["shared_at_polarity"](
                query=torch.nn.functional.normalize(shared_query[-1]).permute(2, 0, 1),
                key=(
                    # key needs some information sharing from target and expression 
                    torch.nn.functional.normalize(polarity_cnn, p=2, dim=-1).permute(2, 0, 1)
                    + target_attn + expression_propagate  # FIXME size problem permute?
                ),
                value=polarity_cnn.permute(2, 0, 1),
                key_padding_mask=((batch[1]*-1)+1).bool().to(torch.device(self.device)),
                need_weights=False
            )
            polarity_inter = shared_query[-1] + polarity_attn.permute(1, 2, 0)
            shared_query.append(polarity_inter)

            # polarity output
            polarity_logits = self.components["polarity"]["linear"](polarity_inter.permute(0, 2, 1))

            # stacking
            target_outputs.append(target_logits.unsqueeze(-1))
            polarity_outputs.append(polarity_logits.unsqueeze(-1))
            expression_outputs.append(expression_logits.unsqueeze(-1))

            # dropout
            target_inter = self.components["target"]["re_encode"](target_inter).permute(0, 2, 1)
            polarity_inter = self.components["polarity"]["re_encode"](polarity_inter.permute(0, 2, 1)).permute(0, 2, 1)
            expression_inter = self.components["expression"]["re_encode"](expression_inter).permute(0, 2, 1)

            # store learning info for next stack
            target_inputs.append(target_inter)
            polarity_inputs.append(polarity_inter)
            expression_inputs.append(expression_inter)


        output = {  # concat across all predictions, take mean of each for the 3 possible labels
            "expression": torch.mean(torch.cat(expression_outputs, dim=-1), dim=-1).permute(0, 2, 1),  # batch, labels, sequence
            "polarity": torch.mean(torch.cat(polarity_outputs, dim=-1), dim=-1).permute(0, 2, 1),  # batch, labels, sequence
            "target": torch.mean(torch.cat(target_outputs, dim=-1), dim=-1).permute(0, 2, 1),  # batch, labels, sequence
        }

        return output

    @staticmethod
    def get_confidence(expression_logits, batch, current_epoch):
        gold_influence = self.get_prob(self.current_epoch, self.find("warm_up_constant", default=5))

        true_expression = batch[2]

        expression_confidence = (  # strengthen signal for true expression_logits
            gold_influence*true_expression.bool().float()
                + (1-gold_influence)*expression_logits.argmax(-1)
        ).detach().to(torch.device(self.device))
        expression_confidence.requires_grad = True
        
        return expression_confidence  # shape: [batch, sequence]


class FgFlex(BertHead):
    """
    A combination of the IMN and RACL setup w/ component flexability.
    Allows experimenter to test different alterations of these setups.

    Parameters:
        TODO Fill in parameters used in self.find()
    """
    def expanding_cnn_block(self, in_channels, out_channels, kernel_size=5, m=2):
        """
        Expands cnn size to m times the original in_channels size, 
        before reducing back down to size out_channels
        """
        return torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Conv1d(
                in_channels = in_channels,
                out_channels = in_channels*m, 
                kernel_size = kernel_size,
                padding = kernel_size//2,
            ),
            torch.nn.Conv1d(
                in_channels = in_channels*m,
                out_channels = out_channels, 
                kernel_size = kernel_size,
                padding = kernel_size//2,
            ),
            torch.nn.ReLU()            
        ).to(torch.device(self.device))

    def cnn_block(self, in_channels, out_channels, kernel_size=5):
        return torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels, 
                kernel_size = kernel_size,
                padding = kernel_size//2,
            ),
            torch.nn.ReLU()            
        ).to(torch.device(self.device))

    def attn_block(self, embed_dim):
        return torch.nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = 1,
            dropout=self.dropout,
        ).to(torch.device(self.device))

    def linear_block(self, in_features, out_features):
        return torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(
                in_features=in_features,
                out_features=out_features
            ), 
            torch.nn.Softmax(dim=-1)
        ).to(torch.device(self.device))

    def init_components(self, subtasks):
        #######################################
        # Potentially unset model params
        #######################################
        # cnn related params 
        cnn_dim = self.find("cnn_dim", default=768)
        expanding_cnn = self.find("expanding_cnn", default=False)
        kernel_size = self.find("kernel_size", default=5)
        
        # configurable expanding cnn block
        cnn_block = self.cnn_block if not expanding_cnn else partial(self.expanding_cnn_block, m=expanding_cnn)
        
        # layers
        shared_layers = self.find("shared_layers", default=1)
        # task-wise layers found during for-loop in "Task-specific CNN layers"

        # learning rates
        attn_lr = self.find("attn_lr", default=self.learning_rate)
        shared_lr = self.find("shared_lr", default=self.learning_rate)
        # relation params        
        stack_count = self.find("stack_count", default=1)
        attention_relations = self.find("attention_relations", 
            default=[
                (first, second)
                for first in subtasks
                for second in subtasks
            ]
        )


        #######################################
        # Base components
        #######################################
        components = torch.nn.ModuleDict({
            task: torch.nn.ModuleDict({
                # final outputs for each task
                "linear": self.linear_block(in_features=cnn_dim, out_features=3),

                # map subtask information back to shared hidden state size
                "re_encode": self.linear_block(
                    in_features=cnn_dim,  # update according to forward
                    out_features=cnn_dim
                )
            })
            for task in subtasks
        })


        #######################################
        # Task-specific CNN layers
        #######################################
        for task in self.subtasks:
            components[task].update(
                {"cnn": torch.nn.Sequential(*[
                    cnn_block(cnn_dim, cnn_dim)
                    for layer in range(self.find(task+"_layers", default=1))
                ])
            })


        ######################################
        # Shared relation components
        ######################################

        # similar to IMN setup
        shared = torch.nn.ModuleDict({
            "cnn_0": self.cnn_block(
                in_channels=768, 
                out_channels=cnn_dim, 
                kernel_size=kernel_size
            ),
            "attn_linear": self.linear_block(cnn_dim*2, 3)
        })
        for layer in range(1, shared_layers):
            shared.update({
                f"cnn_{layer}": self.cnn_block(
                    in_channels=cnn_dim, 
                    out_channels=cnn_dim, 
                    kernel_size=kernel_size
                )
            })
        components.update(torch.nn.ModuleDict({
            "shared": shared
        }))

        # similar to RACL setup
        relations = torch.nn.ModuleDict({
            f"stack_{i}": torch.nn.ModuleDict({
                f"{rel[0]}_at_{rel[1]}": self.attn_block(cnn_dim)
                for rel in attention_relations
            })
            for i in range(stack_count)
        })
        components.update(torch.nn.ModuleDict({
            "relations": relations,
        }))

        if stack_count > 0:
            prev_task = None
            for task in self.subtasks:
                for relation in components["relations"]["stack_0"]:
                    if relation.startswith(task) and (task != prev_task):
                        components[task]["attn_linear"] = self.linear_block(cnn_dim*2, 3)
                        prev_task = task


        # needed to update optimizers w/ interacting components 
        self.other_components = {
            "relations" : {
                "lr": attn_lr
            },
            "shared":{
                "lr": shared_lr
            }
        }

        return components

    def forward(self, batch):
        gold_transmission = self.find("gold_transmission", default=True)
        stack_count = self.find("stack_count", default=1)
        shared_layers = self.find("shared_layers", default=1)
        
        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))
        true_labels = {
            "expression": batch[2].to(torch.device(self.device)),
            "holder": batch[3].to(torch.device(self.device)),
            "polarity": batch[4].to(torch.device(self.device)),
            "target": batch[5].to(torch.device(self.device)),
        }

        embeddings = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        embeddings = self.bert_dropout(embeddings).permute(0, 2, 1)
        sentence_output = embeddings.clone()

        ######################################
        # Shared CNN layers
        ######################################
        shared = self.components["shared"]
        for i in range(shared_layers):
            sentence_output = shared[f"cnn_{i}"](sentence_output)

            # update word embeddings with shared features learned from this cnn layer
            embeddings = torch.cat((embeddings, sentence_output), dim=1) # cat embedding dim

        # only the information learned from shared cnn(s), no embeddings
        initial_shared_features = sentence_output

        ######################################
        # Stacked relation interactions
        ######################################
        task_inputs = {
            task: initial_shared_features
            for task in self.subtasks
        }
        outputs = {
            task: []
            for task in self.subtasks
        }
        attn_outputs = {}
        cnn_outputs = {"shared": initial_shared_features}

        for stack in range(stack_count):
            ######################################
            # Subtask CNN that mimic IMN setup
            ######################################
            for task in self.subtasks:
                task_output = task_inputs[task]
                if "cnn" in self.components[task].keys():
                    cnn_outputs[task] = self.components[task]["cnn"](task_output)
                    # TODO take into account embeddings like in IMN? Then would need to expand linear in init_comp
                    # task_output = torch.cat((embeddings, cnn_outputs[task]), dim=1)  # cat embedding dim
                else:
                    cnn_outputs[task] = task_output

                outputs[task].append(self.components[task]["linear"](task_output.permute(0, 2, 1)))

            cnn_outputs["all"] = sum([cnn_outputs[task] for task in self.subtasks])

            ######################################
            # Subtask relations mimic RACL setup
            ######################################
            prev_first = None
            same_task = []
            for relation in self.components["relations"][f"stack_{stack}"]:
                interacting_tasks = relation.split("_at_")
                first_task = interacting_tasks[0]
                second_task = interacting_tasks[1]

                query = torch.nn.functional.normalize(cnn_outputs[second_task], p=2, dim=-1).permute(2, 0, 1)
                key = torch.nn.functional.normalize(cnn_outputs[first_task], p=2, dim=-1).permute(2, 0, 1)
                value = torch.nn.functional.normalize(cnn_outputs[first_task], p=2, dim=-1).permute(2, 0, 1)  

                mask = ((batch[1]*-1)+1).bool().to(torch.device(self.device))

                # logits transmission
                if gold_transmission and ("all" not in relation) and ("shared" not in relation):
                    first_transmission = self.transmission(outputs[first_task][-1], true_labels[first_task])
                    second_transmission = self.transmission(outputs[second_task][-1], true_labels[second_task])

                    # reshape to match query/key shapes
                    first_transmission = first_transmission.permute(1, 0).unsqueeze(-1).expand(key.shape)
                    second_transmission = second_transmission.permute(1, 0).unsqueeze(-1).expand(query.shape)

                    query = query * second_transmission
                    key = key * first_transmission

                relation_attn, weights = self.components["relations"][f"stack_{stack}"][relation](
                    # expects shape: [seq, batch, cnn_dim]
                    query=query,
                    key=key,
                    value=value,
                    key_padding_mask=mask,
                    need_weights=True
                )

                attn_inter = torch.cat((cnn_outputs[first_task].permute(0, 2, 1), relation_attn.permute(1,0,2)), dim=-1)
                attn_logits = self.components[first_task]["attn_linear"](attn_inter) 

                # store task inputs for next stack, considering multiple relations for first task
                if (first_task == prev_first) or (first_task == "shared") or (prev_first is None):
                    same_task.append(attn_inter)
                else:
                    same_task = [attn_inter]

                task_inputs[first_task] = torch.stack(same_task).mean(dim=0)      

                # stack outputs for this relation (averaged after all stacks complete)
                if first_task == "shared":
                    for task in self.subtasks:
                        outputs[task].append(attn_logits)
                else:
                    outputs[first_task].append(attn_logits)

        # in case no interactions desired
        if stack_count == 0:
            for task in self.subtasks:
                task_output = task_inputs[task]
                if "cnn" in self.components[task].keys():
                    task_output = self.components[task]["cnn"](task_output)
                outputs[task] = [self.components[task]["linear"](task_output.permute(0, 2, 1))]

        final_output = {
            task: torch.stack(outputs[task]).mean(dim=0).permute(0, 2, 1)
            for task in self.subtasks
        }

        return final_output

    @staticmethod
    def get_prob(epoch_count, constant=5):
        """
        Borrowed directly from https://github.com/ruidan/IMN-E2E-ABSA
        
        Compute the probability of using gold opinion labels in opinion transmission
        (To alleviate the problem of unreliable predictions of opinion labels sent from AE to AS,
        in the early stage of training, we use gold labels as prediction with probability 
        that depends on the number of current epoch)

        """
        prob = constant/(constant+torch.exp(torch.tensor(epoch_count).float()/constant))
        return prob

    def transmission(self, logits, true):
        """
        To help guide the attention mechanisms on which tokens to focus more on,
        given logits from individual subtask. 
        """
        # decide how much true labels should influence attention based on current epoch
        gold_influence = self.get_prob(self.current_epoch, self.find("warm_up_constant", default=5))

        focus_scope = (  # strengthen signal from true logits
            gold_influence*true.bool().float()
                + (1-gold_influence)*logits.argmax(-1)
        ).detach().to(torch.device(self.device))
        focus_scope.requires_grad = True
        
        return focus_scope  # shape: [batch, sequence]

