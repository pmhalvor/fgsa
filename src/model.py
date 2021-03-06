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


class BertHead(torch.nn.Module):
    """
    Base flexible multitask learner model developed for this experiment.
    Uses a Bert head w/ linear output. 
    Basically BertSimple, except labels are expected split into subtasks.
    Easily changeable for more complex downstream subtasks
    
    Prediction outputs dictionary of subtask label predictions.

    Parameters: 
        optimizer_name


    """
    def __init__(
        self, 
        bert_finetune=True,
        bert_expression=True,
        bert_holder=True,
        bert_polarity=True,
        bert_target=True,
        bert_path=config.BERT_PATH,  
        device="cuda" if torch.cuda.is_available() else "cpu",
        dropout=0.1,            
        ignore_id=-1,
        loss_function="cross-entropy",  # cross-entropy, dice, mse, or iou 
        lr=1e-7,                
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=2,
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

        # for i in range(5):  # dirty fix to avoid intermittend HTTPError for hugging face
        #     try:
        #         self.bert = BertModel.from_pretrained(bert_path).to(torch.device(self.device))
        #     except Exception as e:
        #         if "HTTPError" in e.msg() and i<4:
        #             continue
        #         raise e 
        #     if self.find("bert") is not None:
        #         break

        # initialize bert head
        self.bert = BertModel.from_pretrained(bert_path).to(torch.device(self.device))


        self.bert.requires_grad = self.finetune
        self.bert_dropout = torch.nn.Dropout(self.dropout).to(torch.device(self.device)) 
        
        # architecture specific components: init_comp returns dict of task-specific output layers
        self.components = torch.nn.ModuleDict(self.init_components(self.subtasks)).to(torch.device(self.device))   

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

    ########### Component blocks ###########
    def attn_block(self, embed_dim):
        return torch.nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = 1,
            dropout=self.dropout,
        ).to(torch.device(self.device))

    def cnn_block(self, in_channels, out_channels, kernel_size=5):
        use_cnn_dropout = self.find("use_cnn_dropout", default=False)
        use_cnn_activation = self.find("use_cnn_activation", default=False)  # TODO true?

        layers = []

        if use_cnn_dropout: 
            layers.append(torch.nn.Dropout(self.dropout))

        layers.append(torch.nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels, 
            kernel_size = kernel_size,
            padding = kernel_size//2,
        ))

        if use_cnn_activation:
            layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers).to(torch.device(self.device))

    def expanding_cnn_block(self, in_channels, out_channels, kernel_size=5, m=2):
        """
        Expands cnn size to m times the original in_channels size, 
        before reducing back down to size out_channels
        """
        use_cnn_dropout = self.find("use_cnn_dropout", default=False)
        use_cnn_activation = self.find("use_cnn_activation", default=False)

        layers = []

        if use_cnn_dropout: 
            layers.append(torch.nn.Dropout(self.dropout))

        layers.append(torch.nn.Conv1d(
            in_channels = in_channels,
            out_channels = in_channels*m, 
            kernel_size = kernel_size,
            padding = kernel_size//2,
        ))
        layers.append(torch.nn.Conv1d(
            in_channels = in_channels*m,
            out_channels = out_channels, 
            kernel_size = kernel_size,
            padding = kernel_size//2,
        ))

        if use_cnn_activation:
            layers.append(torch.nn.ReLU())

        return torch.nn.Sequential(*layers).to(torch.device(self.device))

    def split_cnn_block(self, in_channels, out_channels, kernels=[3, 5], task_layers=1):
        """ 

        Implementation style: 
            1. split into kernel sizes
            2. go through all task layers
            3. concatenate to original output size

        Parameters:
            kernels (list): sizes of kernels wished to split cnn between.
                NOTE: kernel count must be divisible by 768, ex (1,2,3,4,6,8,...)
        """
        use_cnn_dropout = self.find("use_cnn_dropout", default=False)  # FIXME True?
        use_cnn_activation = self.find("use_cnn_activation", default=False)

        splits = torch.nn.ModuleList([])

        for kernel_size in kernels: 
            elements = []


            for task_layer in range(task_layers):

                if use_cnn_dropout: 
                    elements.append(torch.nn.Dropout(self.dropout))

                if task_layer == 0:
                    # first layer expects inputs of orig size
                    elements.append(torch.nn.Conv1d(
                        in_channels = in_channels,
                        out_channels = int(out_channels/len(kernels)), 
                        kernel_size = kernel_size,
                        padding = kernel_size//2,
                    ))
                else:
                    # all other layers expect inputs size as previous output
                    elements.append(torch.nn.Conv1d(
                        in_channels = int(out_channels/len(kernels)),
                        out_channels = int(out_channels/len(kernels)), 
                        kernel_size = kernel_size,
                        padding = kernel_size//2,
                    ))

                if use_cnn_activation:
                    elements.append(torch.nn.ReLU())
                
            splits.append(torch.nn.Sequential(*elements))

        return splits.to(torch.device(self.device))
        
    def linear_block(self, in_features, out_features):
        use_linear_activation = self.find("use_linear_activation", default=False)
        use_linear_dropout = self.find("use_linear_dropout", default=False)  #TODO true?

        layers = []

        if use_linear_dropout:
            layers.append(torch.nn.Dropout(self.dropout))
        
        layers.append(torch.nn.Linear(
            in_features=in_features,
            out_features=out_features
        ))

        if use_linear_activation:
            layers.append(torch.nn.Softmax(dim=-1))

        return torch.nn.Sequential(*layers).to(torch.device(self.device))
    
    ########### Model training ###########
    def fit(self, train_loader, dev_loader=None, epochs=10):
        writer = SummaryWriter()

        for epoch in range(epochs):
            self.current_epoch = epoch

            for b, batch in enumerate(train_loader):
                self.train()        # turn off eval mode

                # feed batch to model
                try:
                    output = self.forward(batch)
                except Exception as e:
                    print(f"Most likely memory error in batch {b} with shape {batch[0].shape}")
                    print(torch.cuda.memory_summary(abbreviated=False))
                    # raise e
                    continue  # skip over batches throwing OOM errors
                    

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

                # free up some memory
                torch.cuda.empty_cache()


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
        # for task in self.subtasks:
        #     retain_graph = True if task != self.subtasks[-1] else False
        #     self.losses[task].backward(retain_graph=retain_graph)  # retain_graph needed to update shared tasks
        loss_total.backward()

        # TODO should there be bert optimizer alone, 
        # if so needs to be updated for each task 
        # in addition to task specific optimizers

        # update weights from the model by calling optimizer.step()
        for task in self.optimizers:
            self.optimizers[task].step()

        return self.losses

    def evaluate(self, loader, verbose=False, final=False):
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

            # NOTE adding average f1 over scores per subtask to metric totals over all batches
            absa_total_over_batches += f_absa
            binary_total_over_batches += (sum([binary[task] for task in self.subtasks])/len(self.subtasks))
            hard_total_over_batches += (sum([hard[task] for task in self.subtasks])/len(self.subtasks))
            macro_total_over_batches += (sum([macro[task] for task in self.subtasks])/len(self.subtasks))
            proportional_total_over_batches += (sum([proportional[task] for task in self.subtasks])/len(self.subtasks))
            span_total_over_batches += (sum([span[task] for task in self.subtasks])/len(self.subtasks))


            if final:
                logging.debug("FINAL Batch {batch}:".format(batch=b))
                logging.debug("FINAL {metric:13}: {score}".format(metric="absa", score=f_absa))
                logging.debug("FINAL {metric:13}: {score}".format(metric="binary", score=(sum([binary[task] for task in self.subtasks])/len(self.subtasks))))
                logging.debug("FINAL {metric:13}: {score}".format(metric="hard", score=(sum([hard[task] for task in self.subtasks])/len(self.subtasks))))
                logging.debug("FINAL {metric:13}: {score}".format(metric="macro", score=(sum([macro[task] for task in self.subtasks])/len(self.subtasks))))
                logging.debug("FINAL {metric:13}: {score}".format(metric="proportional", score=(sum([proportional[task] for task in self.subtasks])/len(self.subtasks))))
                logging.debug("FINAL {metric:13}: {score}".format(metric="span", score=(sum([span[task] for task in self.subtasks])/len(self.subtasks))))


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
        
        if self.find("safe_gold_transmission", default=False):
            # make sure influence from gold tranmission is low for preditions
            transmission_config = self.find("warm_up_contant")
            self.warm_up_constant = 0.1

        outputs = self.forward(batch)

        if self.find("safe_gold_transmission", default=False):
            # reset transmission value to that provided during init
            self.warm_up_constant = transmission_config  

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
            task: self.find(f"bert_{task}", default=True)
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
                    {"params": self.bert.parameters(), "lr":self.find("bert_lr", default=self.learning_rate)}
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

        # make sure non-subtask-specific components are added to optimizers
        if other_components is not None:
            for component in other_components:
                if other_components[component] is not {}:
                    lr = other_components[component].get("lr")
                    lr = lr if lr is not None else self.learning_rate

                    # IMN architecture
                    if component == "shared":
                        tasks = other_components[component].get("tasks")
                        tasks = tasks if tasks is not None else self.subtasks

                        for task in tasks:
                            optimizers[task].add_param_group(
                                {"params": self.components[component].parameters(), "lr":lr}
                            )
                   
                    # RACL architecture
                    elif component == "relations":
                        for stack in self.components["relations"]:
                            for layer in self.components["relations"][stack]:
                                first_task = layer.split("_at_")[0]
                                second_task = layer.split("_at_")[1]

                                if first_task in optimizers.keys():
                                    print("adding {} to optimizer {}".format(layer, first_task))
                                    optimizers[first_task].add_param_group(
                                        {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                    )
                                elif first_task == "shared":
                                    for task in self.subtasks:
                                        print("adding {} to optimizer {}".format(layer, task))
                                        optimizers[task].add_param_group(
                                            {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                        )
                                else:
                                    print("Whoops! Not sure how to optimize for first task in layer", layer)
                                    logging.warning("Whoops! Not sure how to optimize for first task in layer {}".format(layer))

                                if second_task in optimizers.keys() and first_task != second_task and first_task != "shared":
                                    print("adding {} to optimizer {}".format(layer, second_task))
                                    optimizers[second_task].add_param_group(
                                        {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                    )
                                elif second_task == "shared":
                                    for task in self.subtasks:
                                        if task != first_task:  # avoid error of adding params to same optimizer twice
                                            print("adding {} to optimizer {}".format(layer, task))
                                            optimizers[task].add_param_group(
                                                {"params": self.components[component][stack][layer].parameters(), "lr":lr}
                                            )
                                elif first_task == second_task or (first_task == "shared"):
                                    pass  # parameters already added, skip
                                else:
                                    print("Whoops! Not sure how to optimize for second task in layer", layer)
                                    logging.warning("Whoops! Not sure how to optimize for second task in layer {}".format(layer))


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

    Parameters:
        bidirectional (bool): whether lstms should flow in both directions or not (default = False)
        hidden_size (int): number of nodes per lstm layer (default = 768)
        num_layers (int): number of hidden layers in lstm components (default = 3)
        <task>_lstm (int): which tasks to use an lstm on (default = True)
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
        lstm_tasks = {
            task: self.find(f"{task}_lstm", default=True)
            for task in self.subtasks
        }

        components = torch.nn.ModuleDict({
            task: torch.nn.ModuleDict({
                "linear": self.linear_block(
                    in_features=hidden_size*2 if bidirectional else hidden_size,
                    out_features=3,
                ).to(torch.device(self.device))
            })
            for task in subtasks
        })

        # make lstm tasks configurable via json
        for task in self.subtasks:
            if lstm_tasks[task]:
                print(f"Adding lstm layer for {task}")
                # add lstm to subtask's components
                components[task]["lstm"] = torch.nn.LSTM(
                        input_size=768,
                        hidden_size=hidden_size, 
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout,
                        bidirectional=bidirectional, 
                    ).to(torch.device(self.device)),
            else:
                # replace linear to ensure correct in_features size (if bidirectional)
                components[task] = torch.nn.ModuleDict({
                    "linear": self.linear_block(
                        in_features=768,
                        out_features=3,
                    ).to(torch.device(self.device))
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
            states = embeddings

            if "lstm" in self.components[task].keys():
                states, _ = self.components[task]["lstm"](embeddings)
                
            output[task] = self.components[task]["linear"](states).permute(0, 2, 1)


        return output


class IMN(BertHead):
    """
    Similar to the original IMN structure, just built on a BertHead (in PyTorch).

    Parameters:
        cnn_dim (int): number of cnn outputs passed as hidden state (default = 768)
        <task>_layers (int): task-specific layer counts (default = 1)
        interactions (int): number of stacks/interactions model should have (default = 2)
        find_scope (bool): ability to look for scope as an auxiliary task (default = False)
        gold_transmission (bool): ability to use true labels in attention components for first few epochs (default = True)
        scope_lr (float): set scope specific learning rate (default = self.learning_rate)
        shared_lr (float): learning rate for shared components
        query (str): ability to configure attention inputs for more model flexibility
        key (str): ability to configure attention inputs for more model flexibility
        value (str): ability to configure attention inputs for more model flexibility
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
                "lr": self.find("shared_lr", default=self.learning_rate),
                "tasks": self.subtasks
            }
        }

        #######################################
        # Task-specific CNN layers
        #######################################
        layers = [
            # configure dropout (or activation) via use_cnn_dropout=True in json-configs
            self.cnn_block(cnn_dim, cnn_dim, 5)
            for layer in range(target_layers)
        ]
        target_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["target"]["cnn_sequential"] = target_sequential

        layers = [
            # configure dropout (or activation) via use_cnn_dropout=True in json-configs
            self.cnn_block(cnn_dim, cnn_dim, 5)
            for layer in range(polarity_layers)
        ]
        polarity_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["polarity"]["cnn_sequential"] = polarity_sequential
        
        layers = [
            # configure dropout (or activation) via use_cnn_dropout=True in json-configs
            self.cnn_block(cnn_dim, cnn_dim, 5)
            for layer in range(expression_layers)
        ]
        expression_sequential = torch.nn.Sequential(*layers).to(torch.device(self.device))
        components["expression"]["cnn_sequential"] = expression_sequential

        #######################################
        # Task-specific output layers
        #######################################
        components["target"]["linear"] = self.linear_block(
            # configure dropout (or activation) via use_linear_dropout in json-configs
            in_features=int(768+(shared_layers+1)*cnn_dim),  # bert:768 + shared_cnn:(300 + 300) + target_cnn:300
            out_features=3
        ).to(torch.device(self.device))

        components["expression"]["linear"] = self.linear_block(
            in_features=int(768+(shared_layers+1)*cnn_dim), # bert:768 + shared_cnn:(300 + 300) + expression_cnn:300
            out_features=3
        ).to(torch.device(self.device))

        # polarity had attention before linear
        components["polarity"]["attention"] = torch.nn.MultiheadAttention(
            cnn_dim, num_heads=1
        ).to(torch.device(self.device))
        components["polarity"]["linear"] = self.linear_block(
            in_features=int(2*cnn_dim), # initial_shared_features:300 + polarity_cnn:300
            out_features=polarity_labels  # NOTE: SemEval data has neutral and confusing polarities
        ).to(torch.device(self.device))

        
        #######################################
        # Scope predictions
        #######################################

        components["scope"] = torch.nn.ModuleDict({
                # scope finder: shared -> linear 
                "linear": torch.nn.Sequential(  # FIXME delete or retest
                    torch.nn.Linear(
                        in_features=cnn_dim,
                        out_features=2
                    ),
                    torch.nn.Sigmoid()
                ).to(torch.device(self.device))
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
        components["shared"]["re_encode"] = self.linear_block(
                    # sentence_output:cnn_dim + target_output:3 + expression_output:3 + polarity_output:5
                    in_features=int(cnn_dim + 3 + 3 + polarity_labels),  
                    out_features=cnn_dim,
            ).to(torch.device(self.device))

        return components

    def forward(self, batch):
        cnn_dim = self.find("cnn_dim", default=768)
        interactions = self.find("interactions", default=2)
        expression_layers = self.find("expression_layers", default=2)
        polarity_layers = self.find("polarity_layers", default=2)
        shared_layers = self.find("shared_layers", default=2)
        target_layers = self.find("target_layers", default=2)
        # attention params

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


        if self.find("find_scope", default=False):
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

            # get attention inputs 
            queries, keys, values = self.get_attention_inputs(
                target_cnn_output.permute(2, 0, 1),     # sequence, batch, embedding
                expression_cnn_output.permute(2, 0, 1), 
                polarity_cnn_output.permute(2, 0, 1),
                batch
            )

            # attention block
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
        cnn_dim (int): number of cnn outputs passed as hidden state (default = 768)
        gold_transmission (bool): ability to use true labels in attention components for first few epochs (default = True)
        kernel_size (int): kernel size for cnn components 
        stack_count (int): number of stacks/interactions model should have (default = 1)
        warm_up_constant (float): proportion of true labels to influence attention blocks, dependent on current epoch (default = 5)
    """

    def init_components(self, subtasks):
        cnn_dim = self.find("cnn_dim", default=768)
        stack_count = self.find("stack_count", default=1)
        kernel_size = self.find("kernel_size", default=5)

        components = torch.nn.ModuleDict({
            "target": torch.nn.ModuleDict({
                # aspect extraction: cnn -> relu -> matmul w/ expression -> attention -> cat -> linear
                "cnn": torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels = int(cnn_dim),
                        out_channels = int(cnn_dim), 
                        kernel_size = kernel_size,
                        padding=kernel_size//2
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
                        out_features=int(cnn_dim)
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
                        in_channels = int(cnn_dim),
                        out_channels = int(cnn_dim), 
                        kernel_size = kernel_size,
                        padding=kernel_size//2
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
                        out_features=int(cnn_dim)
                    ),
                    torch.nn.AlphaDropout(
                        self.dropout,
                    ),
                ).to(torch.device(self.device))
            }),
            "polarity":torch.nn.ModuleDict({
                # polarity classification: cnn -> relu -> matmul w/ (embedding) -> attention -> cat -> dropout -> linear
                "cnn": torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels = int(cnn_dim),
                        out_channels = int(cnn_dim), 
                        kernel_size = kernel_size,
                        padding=kernel_size//2
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
                        out_features=cnn_dim
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
        components["relations"] = torch.nn.ModuleDict({
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
            })
        components["shared"] = self.cnn_block(768, int(cnn_dim), kernel_size)

        self.other_components = {
            "relations": {
                "lr": self.learning_rate
            },
            "shared": {
                "lr": self.learning_rate
            }
        }

        return components

    def forward(self, batch):
        gold_transmission = self.find("gold_transmission", default=False)
        stack_count = self.find("stack_count", default=1)

        input_ids = batch[0].to(torch.device(self.device))
        attention_mask = batch[1].to(torch.device(self.device))
        true_labels = {
            "expression": batch[2].to(torch.device(self.device)),
            "target": batch[5].to(torch.device(self.device))
        }

        embeddings = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
        ).last_hidden_state
        embeddings = self.bert_dropout(embeddings).permute(0, 2, 1)

        # reshape embeddings to cnn_dim
        embeddings = self.components["shared"](embeddings)

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

            ### Relation: R1  
            # TODO check what actually needs to be normalize & what happens when no normalize (like IMN)
            # MO2A expects shape: [sequence, batch, cnn_dim]
            query_expression = torch.nn.functional.normalize(expression_cnn, p=2, dim=-1).permute(2, 0, 1)
            key_target = torch.nn.functional.normalize(target_cnn, p=2, dim=-1).permute(2, 0, 1)
            value_target = target_cnn.permute(2, 0, 1)
            mask = ((batch[1]*-1)+1).bool().to(torch.device(self.device))

            target_attn, _ = self.components["relations"][f"stack_{i}"]["target_at_expression"](
                query=query_expression,
                key=key_target,
                value=value_target,
                key_padding_mask=mask,
                need_weights=False
            )

            target_inter = torch.cat((target_cnn.permute(0, 2, 1), target_attn.permute(1,0,2)), dim=-1)
            target_logits = self.components["target"]["linear"](target_inter)


            # MA2O expects shape: [sequence, batch, cnn_dim]
            query_target = torch.nn.functional.normalize(target_cnn, p=2, dim=-1).permute(2, 0, 1)
            key_expression = torch.nn.functional.normalize(expression_cnn, p=2, dim=-1).permute(2, 0, 1)
            value_expression = expression_cnn.permute(2, 0, 1)
            # same mask as before

            expression_attn, _ = self.components["relations"][f"stack_{i}"]["expression_at_target"](
                # expects shape: [seq, batch, cnn_dim]
                query=query_target,
                key=key_expression,
                value=value_expression,
                key_padding_mask=mask,
                need_weights=False
            )
            expression_inter = torch.cat((expression_cnn.permute(0, 2, 1), expression_attn.permute(1,0,2)), dim=-1)
            expression_logits = self.components["expression"]["linear"](expression_inter)


            ### Relation: R2 + R3 + R4 
            polarity_cnn = self.components["polarity"]["cnn"](polarity_inputs[-1])

            # [sequence, batch, cnn_dim]
            query_shared = shared_query[-1].permute(2, 0, 1)
            key_polarity = torch.nn.functional.normalize(polarity_cnn, p=2, dim=-1).permute(2, 0, 1)
            value_context = polarity_cnn.permute(2, 0, 1) + target_attn  # R2
            # same mask as before

            if gold_transmission:  # R2 (and R3 if target is added here?)
                expression_transmission = self.transmission(expression_logits, true_labels["expression"])

                value_context += expression_transmission.T.unsqueeze(-1).expand(value_context.shape) 

            # shared-polarity attention
            polarity_attn, _ = self.components["relations"][f"stack_{i}"]["shared_at_polarity"](
                query=query_shared,
                key=key_polarity,
                value=value_context,
                key_padding_mask=mask,
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


class FgFlex(BertHead):
    """
    A combination of the IMN and RACL setup w/ component flexibility.
    Allows experimenter to test different alterations of these setups.

    Parameters:
        attention_relations (list(tuple(str, str))): subtasks desired to feed into attention components (default = None)
        attn_lr (float): ability to specify specific learning rate for attention components (default = self.learning rate)
        cnn_dim (int): number of cnn outputs passed as hidden state (default = 768)
        expanding_cnn (int): ability to expand cnns even larger than normal. Not fully tested (default = None) 
        gold_transmission (bool): ability to use true labels in attention components for first few epochs (default = True)
        kernel_size (int): kernel size for cnn components 
        shared_lr (float): ability to specify specific learning rate for shared components (default = self.learning_rate)
        split_cnn_kernels (list(int)): ability to use multiple kernel sizes for cnn filters (default = None)
        stack_count (int): number of stacks/interactions model should have (default = 1)
        shared_layers (int): number of layers for shared component (default = 1)
        <task>_layers (int): task-specific layer counts (default = 1)
        warm_up_constant (float): proportion of true labels to influence attention blocks, dependent on current epoch (default = 5)
    """
    def init_components(self, subtasks):
        #######################################
        # Potentially unset model params
        #######################################
        # cnn related params 
        cnn_dim = self.find("cnn_dim", default=768)
        expanding_cnn = self.find("expanding_cnn", default=None)
        split_cnn_kernels = self.find("split_cnn_kernels", default=None)
        split_cnn_tasks = self.find("split_cnn_tasks", default=None)
        kernel_size = self.find("kernel_size", default=5)
        
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
        # Task-specific CNN layers
        #######################################
        components = torch.nn.ModuleDict({
            task: torch.nn.ModuleDict({})
            for task in subtasks
        })
        # NOTE: stack can be 0, then task-wise components are skipped, only single linear given
        if stack_count > 0:
            for stack in range(stack_count):
                for task in self.subtasks:
                    task_layers = self.find(task+"_layers", default=1)
                    # NOTE: if task_layers == 0, empty sequential cnn_0 is made, but data still flows through

                    if task_layers == 0:
                        # NOTE empty Sequential gives len == 0 in forward, causing only linear to be used
                        components[task][f"cnn_{stack}"] = torch.nn.Sequential(*[])
                        
                    ## CNN component: three possible cnn types for subtasks
                    if expanding_cnn:  # Sometimesse will be 0, other times None
                        components[task][f"cnn_{stack}"] = torch.nn.Sequential(*(
                            # first layer needs to handle larger sizes from shared-emd-cat
                            [self.expanding_cnn_block((768+cnn_dim), cnn_dim, kernel_size, m=expanding_cnn)]
                            +
                            [
                                self.expanding_cnn_block(cnn_dim, cnn_dim, kernel_size, m=expanding_cnn)
                                for layer in range(1, task_layers)
                            ]
                        ))
                    elif None not in (split_cnn_tasks, split_cnn_kernels): 
                        if task in split_cnn_tasks:
                            # NOTE sequential inside split to preserve kernel-wise cnns
                            components[task][f"cnn_{stack}"] = self.split_cnn_block(
                                (768+cnn_dim), cnn_dim, split_cnn_kernels, task_layers
                            )
                        else:
                            components[task][f"cnn_{stack}"] = torch.nn.Sequential(*(
                                # first layer needs to handle larger sizes from shared-emd-cat
                                [self.cnn_block((768+cnn_dim), cnn_dim, kernel_size)]
                                +
                                [
                                    self.cnn_block(cnn_dim, cnn_dim, kernel_size)
                                    for layer in range(1, task_layers)
                                ]
                            ))
                    else:
                        components[task][f"cnn_{stack}"] = torch.nn.Sequential(*(
                            # first layer needs to handle larger sizes from shared-emd-cat
                            [self.cnn_block((768+cnn_dim), cnn_dim, kernel_size)]
                            +
                            [
                                self.cnn_block(cnn_dim, cnn_dim, kernel_size)
                                for layer in range(1, task_layers)
                            ]
                        ))

                    ## Feedforward component: Expands per stack due to concatentation
                    components[task][f"linear_{stack}"] = self.linear_block(in_features=((768+cnn_dim)), out_features=3)

        else: 
            for task in subtasks:
                # final outputs for each task
                # in_features are concats of embeddings, shared layers, and task cnns
                components[task]["linear"] = self.linear_block(in_features=(768+cnn_dim), out_features=3)
        ######################################
        # Shared relation components
        ######################################

        ## Shared layers: similar to IMN setup
        # default difference: IMN uses split_cnn_block w/ kernel=3 & kernel=5
        if split_cnn_kernels is not None:  # NOTE only splitting on first shared
            shared = torch.nn.ModuleDict({
                "cnn_0": self.split_cnn_block(
                    in_channels=768, 
                    out_channels=cnn_dim, 
                    kernels = split_cnn_kernels
                ),
            })
        else:
            shared = torch.nn.ModuleDict({
                "cnn_0": self.cnn_block(
                    in_channels=768, 
                    out_channels=cnn_dim, 
                    kernel_size=kernel_size
                ),
            })

        for layer in range(1, shared_layers):
            # TODO use split_cnn_blocks in all following layers?
            shared[f"cnn_{layer}"] = self.cnn_block(
                    in_channels=cnn_dim, 
                    out_channels=cnn_dim, 
                    kernel_size=kernel_size
                )
        components["shared"] =  shared

        ## Relation layers: similar to RACL setup
        relations = torch.nn.ModuleDict({})
        attn_linears = torch.nn.ModuleDict({})
        for stack in range(stack_count):
            relations[f"stack_{stack}"] = torch.nn.ModuleDict({})
            for rel in attention_relations:
                if len(rel)<2:
                    continue
                if f"{rel[0]}_at_{rel[1]}" in relations:
                    continue
                relations[f"stack_{stack}"][f"{rel[0]}_at_{rel[1]}"] = torch.nn.ModuleDict({
                    "attn": self.attn_block(cnn_dim),

                    # new linear for each attn relation for first_task logits using attn information
                    "linear": self.linear_block(cnn_dim*2, 3)
                })
                # BUG: if the same relation is added multiple times, only 1 attention block is available, 
                # yet, re-encoding will expand to size included all of same relations. 
                # FIX: Check that first task is same, but second task is different when counting for re-encoder size
                if rel[0] in self.subtasks + ["shared"] and "re_encode" not in components[rel[0]].keys():
                    # only reach this one time per first_task=rel[0]
                    count = sum([rel[0] == r[0] if rel[1] != r[1] else False for r in attention_relations]) + 1  # ugly fix to bug above
                    
                    # map subtask information back to shared hidden state size after (multiple) attention(s)
                    components[rel[0]]["re_encode"] = self.linear_block(
                        in_features=cnn_dim*(1 + count),  # 1 task output + number of relations with rel[0] as first task
                        out_features=(768+cnn_dim)
                    )

        components["relations"] = relations

        # Notify init_optimizers about non-subtask components 
        self.other_components = {
            "relations" : {
                "lr": attn_lr
            },
            "shared":{
                "lr": shared_lr,
                "tasks": self.subtasks
            }
        }

        return components

    def forward(self, batch):
        gold_transmission = self.find("gold_transmission", default=True)
        stack_count = self.find("stack_count", default=1)
        shared_layers = self.find("shared_layers", default=1)
        split_cnn_kernels = self.find("split_cnn_kernels", default=None)
        split_cnn_tasks = self.find("split_cnn_tasks", default=None)
        
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
        shared_output = embeddings

        ######################################
        # Shared CNN layers
        ######################################
        shared = self.components["shared"]

        if split_cnn_kernels is not None:
            # split cnn output is a list
            splits = len(shared[f"cnn_0"])
            shared_output = torch.cat(
                [
                    cnn(shared_output)
                    for cnn in shared[f"cnn_0"]
                ],
                dim=1  # concat on the embedding dimension
            )
        else:
            shared_output = shared[f"cnn_0"](shared_output)

        for i in range(1, shared_layers):
            shared_output = shared[f"cnn_{i}"](shared_output)

        # initial shared features consists of embedding plus shared cnn output
        initial_shared_features = torch.cat((embeddings, shared_output), dim=1) # cat embedding dim

        ######################################
        # Stacked relation interactions
        ######################################
        task_inputs = {  # initial shared features consists of embedding plus shared cnn output
            task: initial_shared_features
            for task in self.subtasks
        }
        outputs = {  # all task-specific outputs stored along the way, averaged at end
            task: []
            for task in self.subtasks
        }
        cnn_outputs = {"shared": shared_output}  # only the information learned from shared cnn(s), no embeddings

        for stack in range(stack_count):
            ######################################
            # Subtask CNN that mimic IMN setup
            ######################################
            for task in self.subtasks:
                task_output = task_inputs[task]

                if split_cnn_tasks is not None and task in split_cnn_tasks: 
                    cnn_outputs[task] = torch.cat(
                        [
                            cnn(task_output)
                            for cnn in self.components[task][f"cnn_{stack}"]
                        ],
                        dim=1
                    )
                elif len(self.components[task][f"cnn_{stack}"]) > 0:
                        
                    cnn_outputs[task] = self.components[task][f"cnn_{stack}"](task_output)
                else: 
                    cnn_outputs[task] = shared_output
                
                # cat embedding dim when task cnn exists (i.e. task_layers not 0)
                task_output = torch.cat((embeddings, cnn_outputs[task]), dim=1)  

                task_output = self.components[task][f"linear_{stack}"](task_output.permute(0, 2, 1))

                outputs[task].append(task_output)

            cnn_outputs["all"] = sum([cnn_outputs[task] for task in self.subtasks])

            ######################################
            # Subtask relations mimic RACL setup
            ######################################
            prev_first = None
            relation_inters = {}
            for relation in self.components["relations"][f"stack_{stack}"]:
                relation_components = self.components["relations"][f"stack_{stack}"][relation]

                # parse relation name formatted first_at_second (first=keys,values and second=query)
                interacting_tasks = relation.split("_at_")
                first_task = interacting_tasks[0]
                second_task = interacting_tasks[1]

                # attention head expects shape: [seq, batch, cnn_dim]
                query = (cnn_outputs[second_task]).permute(2, 0, 1)
                key = (cnn_outputs[first_task]).permute(2, 0, 1)
                value = (cnn_outputs[first_task]).permute(2, 0, 1)  

                mask = ((batch[1]*-1)+1).bool().to(torch.device(self.device))

                # logits transmission
                if gold_transmission and ("all" not in relation) and ("shared" not in relation):
                    first_transmission = self.transmission(
                        # pass information about if subtask found label present
                        outputs[first_task][-1].detach().argmax(-1).bool().float(),  
                        true_labels[first_task].bool().float()
                    ).permute(1,0).unsqueeze(-1).expand(value.shape)
                    # first_transmission = self.transmission(value, true_labels[first_task])
                    # second_transmission = self.transmission(outputs[second_task][-1], true_labels[second_task])

                    # reshape to match query/key shapes
                    # first_transmission = first_transmission.permute(1, 0).unsqueeze(-1).expand(key.shape)
                    # second_transmission = second_transmission.permute(1, 0).unsqueeze(-1).expand(query.shape)

                    # NOTE only apply gold transmission to keys (first task), so values help "remap" to previous state
                    # query = query * second_transmission
                    # key = key * first_transmission
                    value = value + (value * first_transmission)


                relation_attn, weights = relation_components["attn"](
                    # expects shape: [seq, batch, cnn_dim]
                    query=query,
                    key=key,
                    value=value,
                    key_padding_mask=mask,
                    need_weights=True
                )

                attn_inter = torch.cat((cnn_outputs[first_task].permute(0, 2, 1), relation_attn.permute(1,0,2)), dim=-1)  # embedding dim
                attn_logits = relation_components["linear"](attn_inter)  # produce logits for first_task using info from attn

                # stack attention outputs for this relation (averaged after all stacks complete)
                if first_task not in ("shared", "all"):
                    outputs[first_task].append(attn_logits)

                # append attn output to list if first task present, else create new list
                if first_task in relation_inters:
                    relation_inters[first_task].append(relation_attn.permute(1,0,2))
                else:
                    relation_inters[first_task] = [relation_attn.permute(1,0,2)]

            # concatenate all attention outputs using same first task w/ that task's conv outputs
            for task in relation_inters:
                if task in self.subtasks:
                    all_task_inters = torch.cat([cnn_outputs[task].permute(0, 2, 1)] + relation_inters[task], dim=-1)
                    task_inputs[task] = self.components[task]["re_encode"](all_task_inters).permute(0, 2, 1)  # batch, sequence, embedding


        # in case no interactions desired
        if stack_count == 0:
            for task in self.subtasks:
                task_output = task_inputs[task]
                outputs[task] = [self.components[task]["linear"](task_output.permute(0, 2, 1))]

        # every task has updated outputs for every stack (for both conv and attn relations) with logits predictions
        # here, and average of the predictions are taken, producing our final output predictions
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

    def transmission(self, predictions, true):
        """
        To help guide the attention mechanisms on which tokens to focus more on,
        given predictions from individual subtask. 
        """
        # decide how much true labels should influence attention based on current epoch
        gold_influence = self.get_prob(self.current_epoch, self.find("warm_up_constant", default=5))

        focus_scope = (  # strengthen signal from true label
            gold_influence*true
                + (1-gold_influence)*predictions 
        ).detach().to(torch.device(self.device))
        focus_scope.requires_grad = True
        
        return focus_scope  # shape: [batch, sequence]

