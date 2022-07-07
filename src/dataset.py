from typing import List, Tuple
from transformers import BertTokenizer
from torch.utils.data import Dataset

import torch
import logging
import os

### LOCAL 
from fgsa.src.config import *


class Norec(Dataset):
    """
    Base Norec dataset class.

    __get__ return order:
        0: input_ids
        1: attention_mask
        2: expression
        3: holder
        4: polarity
        5: target

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
            # tokenizer.convert_tokens_to_ids(row)  # BUG second tokenize (after preprocessed)
            # tokenizer.encode(row)  # NOTE now a CLS and SEP token are added to the tokenized rows
            tokenizer.tokenize(row)  # NOTE now labels need to be expanded at '##'
            for row in sentences
        ]

        ids = [
            tokenizer.encode(row)  # NOTE now a CLS and SEP token are added to the tokenized rows
            for row in tokenized
        ]
        return tokenized, ids
            

    def __init__(
        self, 
        bert_path=BERT_PATH,
        data_dir=DATA_DIR,
        partition = "train",
        proportion=None, 
        ignore_id=-1,
        tokenizer=None,
        max_sent_len=None,
    ):
        self.tokenizer = self.get_tokenizer(bert_path, tokenizer)
        self.IGNORE_ID = ignore_id  # FIXME get form BertTokenizer, idk if this is FIXME is needed
        self.unk_id = self.tokenizer._convert_token_to_id(self.tokenizer.unk_token)
        self.partition = partition

        # parse raw data
        data_path = os.path.join(data_dir, partition)
        data = self.load_raw_data(data_path)
        self.sentence = data[3]  # only data piece not used in build_labels()

        # tokenize in init for faster get item
        self.tokens, self.ids = self.tokenize(self.sentence, self.tokenizer)

        # build dataset specific labels
        self.label = self.build_labels(data)

        # reduce size for faster dev
        self.shrink(p=proportion)

        # remove large sentences
        self.remove_large_sents(max_sent_len=max_sent_len)

        # sanity check of shapes
        try:
            assert len(self.ids) == len(self.label["target"])   # check number of ids vs targets
            assert len(self.ids[0]) == len(self.label["target"][0])  # check number of ids in first ids vs target
            assert len(self.ids[-1]) == len(self.label["target"][-1])  # check number of ids in last ids vs target
            assert len(self.label.keys()) == 4   # check size of a single token 
        except AssertionError as ex:
            print("len", len(self.ids), len(self.label["target"]))
            print("1st", self.ids[0], self.label["target"][0])
            print("last", self.ids[-1], self.label["target"][-1])
            raise ex


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

        # BUG: part of the double tokenizing preprocessing error
        with open(data_path+'/sentence.txt', encoding='utf-8') as f:  # only needs tokens as strings
            sentence = [line.strip() for line in f.readlines()]

        with open(data_path+'/target.txt') as f:
            target = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        try:  # Some datasets won't have this annotation, and should therefore ignore it. 
            with open(data_path+'/holder.txt') as f:  # NOTE filename opinion -> object name expression 
                holder = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        except FileNotFoundError:
            logging.warning("holder.txt not found at path {}. Generating blank list...".format(data_path))
            holder = [[self.IGNORE_ID for _ in line] for line in target]  # FIXME replace with UNK for bert, or mask?

        return (expression, holder, polarity, sentence, target)

    def shrink(self, p=None):
        if p:
            count = int(len(self.sentence)*p)

            for key, value in self.label.items():
                self.label[key] = value[:count]
            self.sentence = self.sentence[:count]
            self.tokens = self.tokens[:count]
            self.ids = self.ids[:count]

            logging.info("Dataset {part} shrunk by a scale of {p}. Now {c} rows.".format(
                part=self.partition, p=p, c=count)
            )

    def remove_large_sents(self, max_sent_len):
        """  
        Check sentence lengths of expanded token ids, and removes those with 
        longer lengths than para max_sent_len. Helps avoid OOM errors on GPU
        """
        to_remove = []
        if max_sent_len is not None:
            for index, sentence in enumerate(self.ids):
                if len(sentence) > max_sent_len:
                    # add indexes in reverse order so pop works later
                    to_remove.insert(0, index)
        
        for index in to_remove:
            self.ids.pop(index)   
            self.sentence.pop(index) 
            self.label["expression"].pop(index)
            self.label["holder"].pop(index)
            self.label["polarity"].pop(index)
            self.label["target"].pop(index)

    def get_tokenizer(self, bert_path=None, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif bert_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        return self.tokenizer

    def __getitem__(self, index):
        """
        Requires self.labels to be set as dict containing each subtask annotation
        """
        self.index = index

        # self.label.keys() == [expression, holder, polarity, target]
        self.current_expression = self.label["expression"][index]
        self.current_holder = self.label["holder"][index]
        self.current_polarity = self.label["polarity"][index]
        self.current_target = self.label["target"][index]

        # ids for self.tokenizer mapping
        self.input_ids = self.ids[index]

        # list representing size of this sequence
        self.attention_mask = [1 for _ in self.input_ids]

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.attention_mask),
            torch.LongTensor(self.current_expression),
            torch.LongTensor(self.current_holder),
            torch.LongTensor(self.current_polarity),
            torch.LongTensor(self.current_target),
        )

    def __len__(self):
        return len(self.sentence)

    # dependent on dataset type
    def build_labels(self, data):
        """
        Using data loaded in load_raw_data(), this method builds a dictionary containing each annotation.
        
        NOTE: data[3] contains sentences of strings. Index [3] is not used in this method.

        Parameters:
            data (list(list(int))): Should contain raw data returned from load_raw_data().

        """
        expression = []
        holder = []
        polarity = []
        target = []

        # NOTE expand labels according to new tokenization scheme

        for r, (row, token_row)  in enumerate(zip(data[0], self.tokens)):  # get each row of data
            # add placeholder for [CLS] token
            row_expression = [self.IGNORE_ID]  # ignore this token during evaluation
            row_holder = [self.IGNORE_ID]  # ignore this token during evaluation
            row_polarity = [self.IGNORE_ID]  # ignore this token during evaluation
            row_target = [self.IGNORE_ID]  # ignore this token during evaluation

            e = 0  # count how many expanded tokens so far in row

            for i, token in enumerate(token_row):  # get each token in row
                if "##" == token[:2]:
                    # expand in other words, give these labels ignores
                    row_expression.append(self.IGNORE_ID)
                    row_holder.append(self.IGNORE_ID)
                    row_polarity.append(self.IGNORE_ID)
                    row_target.append(self.IGNORE_ID)
                    e+=1
                else:
                    try:
                        row_expression.append(data[0][r][i-e])  # indexes 
                        row_holder.append(data[1][r][i-e])
                        row_polarity.append(data[2][r][i-e])
                        row_target.append(data[4][r][i-e])  # NOTE [4] for targets, not [3] which is sentences
                    except IndexError as ex:
                        row_expression.append(self.IGNORE_ID)
                        row_holder.append(self.IGNORE_ID)
                        row_polarity.append(self.IGNORE_ID)
                        row_target.append(self.IGNORE_ID)


            # add placeholder for [SEP] token
            row_expression += [self.IGNORE_ID]  # ignore this token during evaluation
            row_holder += [self.IGNORE_ID]  # ignore this token during evaluation
            row_polarity += [self.IGNORE_ID]  # ignore this token during evaluation
            row_target += [self.IGNORE_ID]  # ignore this token during evaluation
            
            # add to full lists
            expression.append(row_expression)
            holder.append(row_holder)
            polarity.append(row_polarity)
            target.append(row_target)

        return {
            "expression": expression,
            "holder": holder,
            "polarity": polarity,
            "target": target,
        }
