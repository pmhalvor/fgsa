from typing import List, Tuple
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import logging


class NorecOneHot(Dataset):
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
        for e, h, p, t in zip(expression, holder, target, polarity):
            if e == 1:                  # beginning expression
                encoded.append(1)
            elif e == 2:                # inside expression
                encoded.append(2)
            elif h == 1:                # beginning holder
                encoded.append(3)
            elif h == 2:                # inside holder
                encoded.append(4)
            elif t == 1:                # beginning target
                if p == 1:              # postive  TODO double check 1=positive and 2=negative
                    encoded.append(5)
                elif p == 2:            # negative
                    encoded.append(7)
                else:                   # neutral  NOTE norec_fine should not use this
                    encoded.append(0)
            elif t == 2:                # inside target
                if p == 1:              # postive
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
        bert_path="ltgoslo/norbert",
        data_path="$HOME/data/norec_fine/train",
        proportion=None, 
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
        self.IGNORE_ID = -1  # FIXME might have to be positive? & check encode() -> holder
        # self.BIO_indexer['[MASK]'] = self.IGNORE_ID

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # NOTE opinion -> expression for consistency w/ project description
        data = self.load_raw_data(data_path)
        self.expression = data[0]
        self.holder = data[1]
        self.polarity = data[2]
        self.sentence = data[3]
        self.target = data[4]

        logging.info("sentence:    {}".format(len(self.sentence)))
        logging.info("expression:  {}".format(len(self.expression)))
        logging.info("holder:      {}".format(len(self.holder)))
        logging.info("polarity:    {}".format(len(self.polarity)))
        logging.info("target:      {}".format(len(self.target)))

        self.label = self.one_hot_encode(
            self.expression,
            self.holder,
            self.polarity, 
            self.target,
        )


        logging.info("sentence:      {}".format(len(self.sentence)))
        logging.info("label: {}".format(len(self.label)))
        self.tokenized_sentence, self.expanded_label = self.tokenize(self.sentence, self.label)
        logging.info("tokenized_sentence: {}".format(len(self.tokenized_sentence)))
        logging.info("expanded_label:     {}".format(len(self.expanded_label)))

        if proportion is not None:
            count = int(len(self.sentence)*proportion)
            self.expanded_label = self.expanded_label[:count]
            self.tokenized_sentence = self.tokenized_sentence[:count]
            # below not needed, but ok to have
            self.expression = self.expression[:count]
            self.holder = self.holder[:count]
            self.label = self.label[:count]
            self.polarity =  self.polarity[:count]
            self.sentence = self.sentence[:count]
            self.target = self.target[:count]

        # check shapes
        logging.info("sentence: {}".format(len(self.tokenized_sentence)))
        logging.info("label:    {}".format(len(self.expanded_label)))
        logging.info("sentence: {}".format(len(self.tokenized_sentence[0])))
        logging.info("expanded_label:    {}".format(len(self.expanded_label[0])))
        logging.info("sentence: {}".format(len(self.tokenized_sentence[-1])))
        logging.info("expanded_label:    {}".format(len(self.expanded_label[-1])))
        assert len(self.tokenized_sentence) == len(self.expanded_label)
        assert len(self.tokenized_sentence[0]) == len(self.expanded_label[0])
        assert len(self.tokenized_sentence[-1]) == len(self.expanded_label[-1])


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
            # sentence = [line.strip().split(' ') for line in f.readlines()]
            sentence = [line for line in f.readlines()]

        with open(data_path+'/target.txt') as f:
            target = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        try:  # Some datasets won't have this annotation, and should therefore ignore it. 
            with open(data_path+'/holder.txt') as f:  # NOTE filename opinion -> object name expression 
                holder = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]
        except FileNotFoundError:
            logging.warning("holder.txt not found at path {}. Generating blank list...".format(data_path))
            holder = [[-1 for _ in line] for line in target]  # TODO give ignore index?

        return (expression, holder, polarity, sentence, target)


    def tokenize(self, sentences, labels):
        """
        Parameters:
            sentences: List[str] raw lists from data set
            labels: List[List[int]] one hot encoded labels
        """
        tokenized_sentences = []
        expanded_labels = []

        logging.info("sentences:      {}".format(len(sentences)))
        logging.info("labels: {}".format(len(labels)))

        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            tokenized_sentence = self.tokenizer(sentence)
            tokenized_sentences.append(tokenized_sentence)

            input_ids = tokenized_sentence['input_ids']
            tokens = self.tokenizer.decode(input_ids).strip().split(' ')

            expanded_label = [self.IGNORE_ID]  #  Needs start token 
            unused_label = label.copy()

            logging.info("tokens: \n\t{}".format(tokens))
            logging.info("sentence: \n\t{}".format(sentence))
            logging.info("label: \n\t{}".format(label))

            for i, t in enumerate(tokens):
                """
                Here we need to expand one_hot_labels to correctly match length of tokens.
                """
                if t.startswith("##"):
                    expanded_label.append(expanded_label[-1])
                    # FIXME this is going to create multiple B tags..
                elif i==len(tokens)-2:
                    expanded_label.append(self.IGNORE_ID)  # Needs end token
                    break
                else:
                    logging.info("else unused_label: \t{}".format(unused_label))
                    logging.info("else expanded_label:\t{}".format(expanded_label))
                    logging.info("else label:\t\t\t{}".format(label))
                    expanded_label.append(unused_label.pop(0))

    
            # logging.info("expanded_label: \n\t{}".format(expanded_label))

            expanded_labels.append(expanded_label)

        return tokenized_sentences, expanded_labels


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

        # self.current_sentence = self.sentence[index]
        self.current_label = self.label[index]

        # tokenize sentence
        # self.tokens = self.tokenizer(
        #     self.current_sentence,
        #     is_split_into_words=True,
        # )  # TODO .squeeze(0) ?
        self.tokens = self.tokenized_sentence[index]

        # bert tokenize expands to <start> tok #en #s in sent #ence <end>
        # yet labels are only mapped to full words
        # how should this be solved? 

        # store token info needed for training
        self.input_ids = self.tokens['input_ids']
        self.attention_mask = self.tokens['attention_mask']

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.attention_mask),
            torch.LongTensor(self.current_label),
        )
    
    
    def __len__(self):
        return len(self.sentence)


class Norec(Dataset):
    def __init__(
        self, 
        bert_path="lgtoslo/norbert",
        data_path="$HOME/data/norec_fine/train", 
    ):
        """
        NotImplemented!
        Use NorecOneHot until this is built.
        
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.sents, self.all_labels = self.load_raw_data(data_path)

        self.indexer = {
            "O": 0,
            "B-targ-Positive": 1,
            "I-targ-Positive": 2,
            "B-targ-Negative": 3,
            "I-targ-Negative": 4,
            "I-targ-Negative": 4,
            "I-targ-Negative": 4,
            "I-targ-Negative": 4,
        }

        self.BIO_indexer = {
            "O": 0,
            "I": 1,
            "B": 2,
        }

        self.polarity_indexer = {
            "O": 0,
            "Positive": 1,
            "Negative": 2,
        }

        self.IGNORE_ID = len(self.BIO_indexer)
        self.BIO_indexer['[MASK]'] = self.IGNORE_ID


    def load_raw_data(self, data_path):
        """
        Load data into Python objects.
        Expected data dir structure:
            norec/
                train/
                    opinion.txt:
                        0 0 0 0 0 1 0 0
                        0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0
                        ...
                    sentence.txt:
                        but the staff was so horrible to us
                        to be completely fair , the only redeeming factor was the food , which was above average , but could n't make up for all the other deficiencies of teodora
                        ...
                    target.txt:
                        0 0 1 0 0 0 0 0
                        0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        ...
                    target_polarity.txt
                        0 0 2 0 0 0 0 0
                        0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        ...

                test/ 
                    ...
                dev/
                    ...
        """
        opinion, sentence, polarity, target = [], [], [], []
        
        data_path = data_path[:-1] if data_path[-1] == '/' else data_path

        with open(data_path+'/sentence.txt') as f:
            sentence = f.readlines()
            self.sentence = sentence

        with open(data_path+'/opinion.txt') as f:
            opinion = f.readlines()
            self.opinion = opinion

        with open(data_path+'/polarity.txt') as f:
            polarity = f.readlines()
            self.polarity = polarity

        with open(data_path+'/target.txt') as f:
            target = f.readlines()
            self.target = target

        return opinion, sentence, polarity, target


    def __getitem__(self, index):
        self.index = index

        self.current_sentence = self.sentence[index]
        self.current_target = self.target[index]
        self.current_polarity = self.polarity[index]
        self.current_opinion = self.opinion[index]

        self.tokens = self.tokenizer(
            self.current_sentence,
            is_split_into_words=True,
        )  # TODO .squeeze(0) ?

        self.input_ids = self.tokens['input_ids']
        self.attention_mask = self.tokens['attention_mask']

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.attention_mask),
            torch.LongTensor(self.current_target),
            torch.LongTensor(self.current_polarity),
            torch.LongTensor(self.current_opinion),
        )
    
    
    def __len__(self):
        return len(self.sentence)

