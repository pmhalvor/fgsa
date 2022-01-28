from typing import List, Tuple
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import logging


class OurDataset(Dataset):

    @staticmethod
    def load_raw_data(data_file):
        """
        """
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file, encoding="ISO-8859-1"):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)


        # sents = [' '.join(s) for s in sents]
        return sents, all_labels

    @staticmethod  # TODO delete or use. NotImplemented
    def getting_y(specify_y, all_labels):

        if specify_y == 'BIO':
            BIO_labels = []
            for row in all_labels:
                bio = []
                for label in row:
                    bio.append(label.split('-')[0])
                BIO_labels.append(bio)

            return BIO_labels, []

        elif specify_y == 'polarity':
            polarity_labels = []
            for row in all_labels:
                polarity = []
                for label in row:
                    if label == 'O':
                        polarity.append(label)
                    else:
                        polarity.append(label.split('-')[2])
                polarity_labels.append(polarity)

            return [], polarity_labels

        elif specify_y == 'both':
            BIO_labels, polarity_labels = [], []
            for row in all_labels:
                bio, polarity = [], []
                for label in row:
                    if label == 'O':
                        polarity.append(label)
                    else:
                        polarity.append(label.split('-')[2])
                    bio.append(label.split('-')[0])
                BIO_labels.append(bio)
                polarity_labels.append(polarity)

            return BIO_labels, polarity_labels
        
        return all_labels

    def __init__(
        self, 
        data_file, 
        specify_y=None, 
        bert_path="lgtoslo/norbert",
        tokenizer=None
    ):
        """
        data_file = 'exam/data/train.conll'
        specify_y = None, 'BIO' or 'polarity' or 'both'
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer 
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.sents, self.all_labels = self.load_raw_data(data_file)
        self.specify_y = specify_y

        if self.specify_y:
            self.BIO_labels, self.polarity_labels = self.getting_y(
                specify_y=self.specify_y,
                all_labels=self.all_labels
            )

        self.indexer = {
            "O": 0,
            "B-targ-Positive": 1,
            "I-targ-Positive": 2,
            "B-targ-Negative": 3,
            "I-targ-Negative": 4
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

        if self.specify_y == 'BIO':
            self.IGNORE_ID = len(self.BIO_indexer)
            self.BIO_indexer['[MASK]'] = self.IGNORE_ID
        elif self.specify_y == 'polarity':
            self.IGNORE_ID = len(self.polarity_indexer)
            self.polarity_indexer['[MASK]'] = self.IGNORE_ID
        elif self.specify_y == 'both':
            self.IGNORE_ID = len(self.BIO_indexer)
            self.BIO_indexer['[MASK]'] = self.IGNORE_ID
            self.polarity_indexer['[MASK]'] = self.IGNORE_ID
        else: #if self.specify_y is None:
            self.IGNORE_ID = len(self.indexer)
            self.indexer['[MASK]'] = self.IGNORE_ID

    def old__getitem__(self, index):
        self.index = index

        self.current_sentence1 = self.sents[index]
        self.current_label = self.all_labels[index]

        # #################### filters
        new_current_sentence = []
        for idx, word in enumerate(self.current_sentence1):
            if len(word) > 1:
                new_word = word.replace("-", "")
                new_word = new_word.replace('—', "")
                new_word = new_word.replace("&", "")
                new_word = new_word.replace("*", "")
                new_word = new_word.replace("@", "")
                new_word = new_word.replace("+", "")
                new_word = new_word.replace("(", "")
                new_word = new_word.replace(")", "")
                new_word = new_word.replace("_", "")
                new_word = new_word.replace("'", "")
                new_word = new_word.replace(".", "")
                new_word = new_word.replace(":", "")
                new_word = new_word.replace("...", "")
                new_word = new_word.replace("/", "")
                new_word = new_word.replace(",", "")
                new_word = new_word.replace("|", "")
                new_current_sentence.append(new_word)
            else:
                new_current_sentence.append(word)

        self.current_sentence = new_current_sentence
        # ####################

        self.input_ids = self.tokenizer(
            self.sents[index],
            is_split_into_words=True,
            #return_tensors='pt'
        )['input_ids']#.squeeze(0) #NOTE: needed squeeze here

        self.attention_mask = self.tokenizer(
            self.sents[index],
            is_split_into_words=True,
            #return_tensors='pt'
        )['attention_mask']#.squeeze(0) #NOTE: needed squeeze here

        if self.specify_y == 'BIO':

            y_BIO = [
                self.BIO_indexer[bt]
                for bt in self.BIO_labels[index]
            ]

            self.y_masks = self._build_y_masks(self.input_ids)
            self.y_extended = self._extend_labels(y_BIO,
                                                  self.y_masks)

        elif self.specify_y == 'polarity':

            y_polarity = [
                self.polarity_indexer[bt]
                for bt in self.polarity_labels[index]
            ]

            return (
                torch.LongTensor(self.input_ids),
                torch.LongTensor(y_polarity),
                torch.LongTensor(self.attention_mask)
            )
            
        elif self.specify_y == 'both':
            # print(self.current_label)
            self.y_BIO = [
                self.BIO_indexer[bt.split('-')[0]]
                for bt in self.current_label
            ]

            self.y_polarity = [
                self.polarity_indexer[pt.split('-')[-1]]
                for pt in self.current_label
            ]
            # print('-'*15)
            # print(self.y_BIO)
            # print(self.y_polarity)
            # print('-'*15)

            # ymask will have size input_ids.shape
            self.y_masks = self._build_y_masks(self.input_ids)
            # y_ext will have size self.current_sentence.split(' ').shape
            self.y_BIO_extended = self._extend_labels(self.y_BIO, self.y_masks)
            self.y_polarity_extended = self._extend_labels(self.y_polarity, self.y_masks)
            # print(
            #     self.y_BIO_extended,
            #     self.y_polarity_extended
            # )
            
            self.y_extended = torch.cat((self.y_BIO_extended, self.y_polarity_extended))
            # self.y_extended = self.y_BIO_extended + self.y_polarity_extended
            # print(len(self.y_BIO_extended))
            # print(len(self.y_polarity_extended))
            # print(len(self.y_extended))

        elif self.specify_y is None:
            self.y = [
                self.indexer[bt]
                for bt in self.current_label
            ]

            self.y_masks = self._build_y_masks(self.input_ids)
            self.y_extended = self._extend_labels(self.y, self.y_masks)

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.y_extended),
            torch.LongTensor(self.attention_mask)
        )

    def __len__(self):
        return len(self.sents)

    def _build_y_masks(self, ids):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~
        train_dataset.current_label >> size >> 17
        ['B-targ-Negative', 'I-targ-Negative', 'I-targ-Negative', 'O',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        train_dataset.current_sentence >> size >> 17
        ['Hiphop-acts', 'med', 'liveband', 'feiler', 'desverre',
         'altfor', 'ofte', '-', 'og', 'dette', 'er', 'et', 'godt',
         'eksempel', 'akkurat', 'dét', '.']

        token and y_mask
        ['[CLS]' = 0, 'Hiphop-acts' = 1, 'med' = 1, 'live' = 1, '##band' = 0,
        'feil' = 1, '##er' = 0, 'des' = 1, '##ver' = 0, '##re' = 0,
        'altfor' = 1, 'ofte' = 1, '-' = 1, 'og' = 1, 'dette' = 1,
        'er' = 1, 'et' = 1, 'godt' = 1, 'eksempel' = 1, 'akkurat' = 1,
        'd' = 1, '##ét' = 0, '.' = 1, '[SEP]=0']
        """
        self.tok_sent = [self.tokenizer.convert_ids_to_tokens(i) for i in ids]
        mask = torch.empty(len(self.tok_sent), dtype=torch.long)

        for i, token in enumerate(self.tok_sent):
            if token.startswith('##'):
                mask[i] = 0
            elif token in self.tokenizer.all_special_tokens + ['[MASK]']:
                mask[i] = 0
            else:
                mask[i] = 1

        return mask

    def _extend_labels(self, labels, mask):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~

        - Sentence:
        'Nominasjonskampen i Oslo SV mellom Heikki Holmås og
        Akhtar Chaudhry i desember i fjor handlet blant annet om
        beskyldninger om juks.'

        - Token and its ID or mask, respectively:
        '[CLS]'=6, 'No'=2, '##min'=0, '##asjons'=0, '##kampen'=0, 'i'=2,
        'Oslo'=3, 'SV'=4, 'mellom'=2, 'Hei'=5, '##kk'=0, '##i'=0,
        'Holm'=1, '##ås'=0, 'og'=2, 'Ak'=5, '##htar'=0, 'Ch'=1,
        '##aud'=0, '##hr'=0, '##y'=0, 'i'=2, 'desember'=2, 'i'=2,
        'fjor'=2, 'handlet'=2, 'blant'=2, 'annet'=2, 'om'=2,
        'beskyld'=2, '##ninger'=0, 'om'=2, 'juks'=2, '##.'=6,
        '[SEP]'=6, '[PAD]'=6 ...

        - returns: torch.tensor containing the token's ID or mask
        tensor([0, 2, 0, 0, 0, 2, 3, 4, 2, 5, 0, 0, 1, 0, 2, 5, 0, 1,
        0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0])
        """
        extended = torch.empty(mask.size(0), dtype=torch.long)

        label_idx = 0
        for i, m in enumerate(mask.tolist()):

            if m == 1 and label_idx<len(labels):
                extended[i] = labels[label_idx]
                label_idx += 1

            else:
                extended[i] = self.IGNORE_ID

        return extended


class NorecOneHot(Dataset):
    def __init__(
        self, 
        bert_path="lgtoslo/norbert",
        data_path="$HOME/data/norec_fine/train", 
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
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        # NOTE opinion -> expression for consistency w/ project description
        self.expression, self.holder, self.sentence, self.polarity, self.target = self.load_raw_data(data_path)

        self.labels = self.one_hot_encode(
            self.expression,
            self.holder,
            self.polarity, 
            self.target,
        )

        self.IGNORE_ID = -1  # FIXME might have to be positive? & check encode() -> holder
        # self.BIO_indexer['[MASK]'] = self.IGNORE_ID

    @staticmethod
    def load_raw_data(data_path) -> Tuple[List,List,List,List]:
        """
        Load data into Python objects.

        Parameters:
            data_path (str): path to data dir. See Misc. below for dir. structure

        Returns:
            expression, holder, sentence, polarity, target

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

        with open(data_path+'/polarity.txt') as f:
            polarity = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/sentence.txt') as f:
            sentence = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        with open(data_path+'/target.txt') as f:
            target = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]

        try:  # Some datasets won't have this annotation, and should therefore ignore it. 
            with open(data_path+'/holder.txt') as f:  # NOTE filename opinion -> object name expression 
                expression = [[int(ele) for ele in line.strip().split(' ')] for line in f.readlines()]
        except FileNotFoundError:
            logging.warning("holder.txt not found at path {}. Generating blank list...".format(data_path))
            holder = [[-1 for _ in line] for line in target]  # TODO give ignore index?

        return expression, holder, sentence, polarity, target


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
        return encoded


    def one_hot_encode(
        self,
        expression, 
        holder,
        polarity,
        target,
    ):
        self.labels = [
            self.encode(e, h, p, t)
            for e, h, p, t in zip(expression, holder, polarity, target)
        ]
        return self.labels


    def __getitem__(self, index):
        self.index = index

        self.current_sentence = self.sentence[index]
        self.current_label = self.labels[index]

        # tokenize sentence
        self.tokens = self.tokenizer(
            self.current_sentence,
            is_split_into_words=True,
        )  # TODO .squeeze(0) ?

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

