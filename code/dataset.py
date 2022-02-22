from typing import List, Tuple
from transformers import BertTokenizer
from torch.utils.data import Dataset

import torch
import logging
import os

### LOCAL 
from config import BERT_PATH
from config import DATA_DIR


class Norec(Dataset):
    """
    Base Norec dataset class.

    Labels order:
        0: expression
        1: holder
        2: polarity
        3: target

    Parameters:
        bert_path (str): location of BertModel for tokenizer
        data_dir (str): directory the preprocessed data is
        partititon (str): train, test, dev set
        proportion (float): proportion of data to load (for development only)
        ignore_id (int): id to assign tokens unknown to tokenizer
        tokenizer (transformers.BertTokenizer): to use same tokenizer between datasets
    """
    @staticmethod
    def tokenize(sentences, tokenizer):
        tokenized = [
            tokenizer.convert_tokens_to_ids(row)
            for row in sentences
        ]
        return tokenized
            

    def __init__(
        self, 
        bert_path=BERT_PATH,
        data_dir=DATA_DIR,
        partition = "train",
        proportion=None, 
        ignore_id=-1,
        tokenizer=None,
    ):
        self.tokenizer = self.get_tokenizer(bert_path, tokenizer)
        self.IGNORE_ID = ignore_id  # FIXME get form BertTokenizer

        # parse raw data
        data_path = os.path.join(data_dir, partition)
        data = self.load_raw_data(data_path)
        self.expression = data[0]
        self.holder = data[1]
        self.polarity = data[2]
        self.sentence = data[3]
        self.target = data[4]

        # build dataset specific labels
        self.label = self.build_labels(
            self.expression,
            self.holder,
            self.polarity,
            self.target,
        )

        # tokenize in init for faster get item
        self.tokens = self.tokenize(self.sentence, self.tokenizer)

        # reduce size for faster dev
        self.shrink(p=proportion)

        # check shapes
        assert len(self.sentence) == len(self.label)   # check number of sentences vs labels
        assert len(self.sentence[0]) == len(self.label[0])  # check number of tokens in first sentence vs label
        assert len(self.sentence[-1]) == len(self.label[-1])  # check number of tokens in last sentence vs label
        assert len(self.label[0][0]) == 4   # check size of a single token 

    def load_raw_data(self, data_path) -> Tuple[List,List,List,List,List]:
        """
        Load data into Python objects.

        Parameters:
            data_path (str): path to data dir. See Misc. below for dir. structure

        Returns:
            (expression, holder, sentence, polarity, target, tokenzied_sentence)

        Misc:
            Expected data dir. structure:
                norec/
                    train/
                        holder.txt:
                            0 0 0 0 0 1 0 0
                            ...
                        opinion.txt:
                            0 0 0 0 0 1 0 0
                            ...
                        sentence.txt:
                            but the staff was so horrible to us
                            ...
                        target.txt:
                            0 0 1 0 0 0 0 0
                            ...
                        target_polarity.txt
                            0 0 2 0 0 0 0 0
                            ...
                    test/ 
                        ...
                    dev/
                        ...
        """
        expression, holder, sentence, polarity, target = [], [], [], [], []
        
        data_path = data_path[:-1] if data_path[-1] == '/' else data_path  # handle both directory expressions

        with open(data_path+'/opinion.txt') as f:  # NOTE file name: opinion -> object name: expression 
            expression = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/target_polarity.txt') as f:
            polarity = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/sentence.txt', encoding='utf-8') as f:  # only needs tokens as strings
            sentence = [line.strip().split(' ') for line in f.readlines()]

        with open(data_path+'/target.txt') as f:
            target = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        try:  # Some datasets won't have this annotation, and should therefore ignore it. 
            with open(data_path+'/holder.txt') as f:  # NOTE filename opinion -> object name expression 
                holder = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]
        except FileNotFoundError:
            logging.warning("holder.txt not found at path {}. Generating blank list...".format(data_path))
            holder = [[self.IGNORE_ID for _ in line] for line in target]  # TODO give ignore index?

        return (expression, holder, polarity, sentence, target)

    def shrink(self, p):
        if p is not None:
            count = int(len(self.sentence)*p)

            self.label = self.label[:count]
            self.sentence = self.sentence[:count]

            # below not needed, but ok to have
            self.polarity =  self.polarity[:count]
            self.target = self.target[:count]

        logging.info("Dataset shrunk by a scale of {p}. Now {c} rows.".format(p=p, c=count))

    def get_tokenizer(self, bert_path=None, tokenizer=None):
        if bert_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        elif tokenizer is not None:
            self.tokenizer = tokenizer
        return self.tokenizer

    def __getitem__(self, index):
        """
        Requires self.labels to be set.
        """
        self.index = index

        # label: e, h, p, t
        self.current_label = self.label[index]

        # ids for self.tokenizer mapping
        self.input_ids = self.tokens[index]

        # list representing size of this sequence
        self.attention_mask = [1 for _ in self.input_ids]

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.attention_mask),
            torch.LongTensor(self.current_label),
        )

    def __len__(self):
        return len(self.sentence)

    # dependent on dataset type
    def build_labels(self, expression, holder, polarity, target):
        label = []

        for r, row  in enumerate(target):  # get each row of data
            encoded_row = []
            for i, _ in enumerate(row):  # get each token in row
                token = [
                    expression[r][i],
                    holder[r][i],
                    polarity[r][i],
                    target[r][i],
                ]
                encoded_row.append(token)
            label.append(encoded_row)

        return label

class NorecOneHot(Norec):
    @staticmethod
    def encode(expression, holder, polarity, target) -> List:
        """
        Handmade one-hot encoder.

          Value       Label
            0           O                       \n
            1       B-Expression                \n
            2       I-Expression                \n
            3       B-Holder                    \n
            4       I-Holder                    \n
            5       B-Target-Positive           \n
            6       I-Target-Positive           \n
            7       B-Target-Negative           \n
            8       I-Target-Negative           \n

        """
        encoded = []
        for e, h, p, t in zip(expression, holder, polarity, target):
            if e == 1:                  # beginning expression
                encoded.append(1)
            elif e == 2:                # inside expression
                encoded.append(2)
            elif h == 1:                # beginning holder
                encoded.append(3)
            elif h == 2:                # inside holder
                encoded.append(4)
            elif t == 1:                # beginning target
                if p == 1:              # positive  TODO double check 1=positive and 2=negative
                    encoded.append(5)
                elif p == 2:            # negative
                    encoded.append(7)
                else:                   # neutral  NOTE norec_fine should not use this
                    encoded.append(0)
            elif t == 2:                # inside target
                if p == 1:              # positive
                    encoded.append(6)
                elif p == 2:            # negative
                    encoded.append(8)
                else:                   # neutral 
                    encoded.append(0)
            else:                       # outside everything
                encoded.append(0)

        assert len(encoded) == len(expression)
        return encoded

    def __init__(
        self, 
        bert_path=BERT_PATH,
        data_path=DATA_DIR + "train",
        proportion=None, 
        ignore_id=-1,
        tokenizer=None,
    ):
        """
        One-hot encoded labels means BIO-tags for target, holder, expression represented 
        together with target polarity.

        NOTE: Intensity is not checked here (yet)

          Value       Label
            0           O                       \n
            1       B-Expression                \n
            2       I-Expression                \n
            3       B-Holder                    \n
            4       I-Holder                    \n
            5       B-Target-Positive           \n
            6       I-Target-Positive           \n
            7       B-Target-Negative           \n
            8       I-Target-Negative           \n

        
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.IGNORE_ID = ignore_id  # FIXME get form BertTokenizer

        # NOTE opinion -> expression for consistency w/ project description
        data = self.load_raw_data(data_path)
        self.expression = data[0]
        self.holder = data[1]
        self.polarity = data[2]
        self.sentence = data[3]
        self.target = data[4]


        self.label = self.one_hot_encode(
            self.expression,
            self.holder,
            self.polarity, 
            self.target,
        )

        if proportion is not None:
            count = int(len(self.sentence)*proportion)

            self.label = self.label[:count]
            self.sentence = self.sentence[:count]

            # below not needed, but ok to have
            self.expression = self.expression[:count]
            self.holder = self.holder[:count]
            self.polarity =  self.polarity[:count]
            self.target = self.target[:count]

        # check shapes
        assert len(self.sentence) == len(self.label)
        assert len(self.sentence[0]) == len(self.label[0])
        assert len(self.sentence[-1]) == len(self.label[-1])

    def load_raw_data(self, data_path) -> Tuple[List,List,List,List,List]:
        """
        Load data into Python objects.

        Parameters:
            data_path (str): path to data dir. See Misc. below for dir. structure

        Returns:
            (expression, holder, sentence, polarity, target, tokenzied_sentence)

        Misc:
            Expected data dir. structure:
                norec/
                    train/
                        holder.txt:     # TODO implement this in preprocessing
                            0 0 0 0 0 1 0 0
                            ...
                        opinion.txt:
                            0 0 0 0 0 1 0 0
                            ...
                        sentence.txt:
                            but the staff was so horrible to us
                            ...
                        target.txt:
                            0 0 1 0 0 0 0 0
                            ...
                        target_polarity.txt
                            0 0 2 0 0 0 0 0
                            ...
                    test/ 
                        ...
                    dev/
                        ...
        """
        expression, holder, sentence, polarity, target = [], [], [], [], []
        
        data_path = data_path[:-1] if data_path[-1] == '/' else data_path  # handle both directory expressions

        with open(data_path+'/opinion.txt') as f:  # NOTE file name: opinion -> object name: expression 
            expression = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/target_polarity.txt') as f:
            polarity = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/sentence.txt') as f:  # only needs tokens as strings
            sentence = [line.strip().split(' ') for line in f.readlines()]

        with open(data_path+'/target.txt') as f:
            target = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        try:  # Some datasets won't have this annotation, and should therefore ignore it. 
            with open(data_path+'/holder.txt') as f:  # NOTE filename opinion -> object name expression 
                holder = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]
        except FileNotFoundError:
            logging.warning("holder.txt not found at path {}. Generating blank list...".format(data_path))
            holder = [[self.IGNORE_ID for _ in line] for line in target]  # TODO give ignore index?

        return (expression, holder, polarity, sentence, target)

    def one_hot_encode(
        self,
        expression, 
        holder,
        polarity,
        target,
    ):
        one_hot_label = [
            self.encode(e, h, p, t)
            for e, h, p, t in zip(expression, holder, polarity, target)
        ]
        return one_hot_label

    def __getitem__(self, index):
        self.index = index

        self.current_label = self.label[index]

        self.tokens = self.sentence[index]

        # store token info needed for training
        self.input_ids = self.tokenizer.convert_tokens_to_ids(self.tokens)  # FIXME move to raw data
        self.attention_mask = [1 for _ in self.input_ids]

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.attention_mask),
            torch.LongTensor(self.current_label),
        )
    

class NorecTarget(NorecOneHot):
    @staticmethod
    def encode(expression, holder, polarity, target) -> List:
        """
        Handmade one-hot encoder.

          Value       Label
            0           O                \n
            1       B-Positive           \n
            2       I-Positive           \n
            3       B-Negative           \n
            4       I-Negative           \n

        """
        encoded = []
        for p, t in zip(polarity, target):
            if t == 1:                  # beginning target
                if p == 1:              # positive  TODO double check 1=positive and 2=negative
                    encoded.append(1)
                elif p == 2:            # negative
                    encoded.append(3)
                else:                   # neutral  NOTE norec_fine should not use this
                    encoded.append(0)
            elif t == 2:                # inside target
                if p == 1:              # positive
                    encoded.append(2)
                elif p == 2:            # negative
                    encoded.append(4)
                else:                   # neutral 
                    encoded.append(0)
            else:                       # outside everything
                encoded.append(0)

        assert len(encoded) == len(target)

        return encoded

    def __init__(
        self, 
        bert_path=BERT_PATH,
        data_path=DATA_DIR + "train",
        proportion=None, 
        ignore_id=-1,
        tokenizer=None,
    ):
        """
        Dataset object that only gives targets and polarities. 
        Built to isolate tasks, checking that model isn't too complex to learn. 

        NOTE: Intensity is not checked here (yet)

          Value       Label
            0           O                \n
            1       B-Positive           \n
            2       I-Positive           \n
            3       B-Negative           \n
            4       I-Negative           \n

        
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.IGNORE_ID = ignore_id  # FIXME get form BertTokenizer

        data = self.load_raw_data(data_path)
        self.polarity = data[2]
        self.sentence = data[3]
        self.target = data[4]

        self.label, self.sentence = self.one_hot_encode(
            [0 for row in self.target for _ in row],  # for lazy inheritance
            [0 for row in self.target for _ in row],  # for lazy inheritance
            self.polarity, 
            self.target,
            self.sentence,
        )

        if proportion is not None:
            count = int(len(self.sentence)*proportion)

            self.label = self.label[:count]
            self.sentence = self.sentence[:count]

            # below not needed, but ok to have
            self.polarity =  self.polarity[:count]
            self.target = self.target[:count]

        # check shapes
        assert len(self.sentence) == len(self.label)
        assert len(self.sentence[0]) == len(self.label[0])
        assert len(self.sentence[-1]) == len(self.label[-1])

    def one_hot_encode(
        self,
        expression, 
        holder,
        polarity,
        target,
        sentence
    ):
        one_hot_label = []
        used_sentence = []
        for e, h, p, t, s in zip(expression, holder, polarity, target, sentence):

            # only use data points where targets are present
            if sum(t)>0: 
                one_hot_label.append(
                    self.encode(e, h, p, t) 
                )   
                used_sentence.append(s)
            
        return one_hot_label, used_sentence
