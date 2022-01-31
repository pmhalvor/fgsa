"""
These methods were borrowed from norec_fine repo.
Found at: https://github.com/ltgoslo/norec_fine/convert_to_bio.py
Copied here for completeness. 
"""
# from nltk import word_tokenize
import config
from transformers import BertTokenizer
from transformers import WordpieceTokenizer

tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)
# tokenizer = WordpieceTokenizer.from_pretrained(config.BERT_PATH)

def word_tokenize(text):
    tokens = tokenizer.tokenize(text, is_split_into_words=True)  # ['input_ids']

    # tokens = tokenizer.decode(input_ids).strip().split(' ')[1:-1]
    return tokens

    # BUG this now forces punctuation to closest binding word,
    # yet some annotations requires punctuation to be alone.
    # What to do?
    #   Skip those annotations?
    #   Force BERT to separate punctuation (how the fuck?)
    #   Try not using is_split_as_words
    #   Try using tokenizer.tokenize() -> tokenizer.encode() -> input_ids
    #   Some unseen answer...?



def get_bio_target(opinion):
    try:
        text, idxs = opinion["Target"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["Polarity"]
            target_tokens = word_tokenize(t)  # NOTE changed from t.split() by pmhalvor
            label = "-targ-{0}".format(polarity)
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    elif not idxs:
        return [(None, None)]
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = word_tokenize(text[0])  # NOTE changed from text[0].split() by pmhalvor
        label = "-targ-{0}".format(polarity)

        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]


def get_bio_expression(opinion):
    try:
        text, idxs = opinion["Polar_expression"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        for t, idx in zip(text, idxs):

            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            polarity = opinion["Polarity"]
            target_tokens = word_tokenize(t)   # NOTE changed from t.split() by pmhalvor
            label = "-exp-{0}".format(polarity)
            tags = []

            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    elif not idxs:  # NOTE elif clause added by pmhalvor 
        return [(None, None)]    
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        polarity = opinion["Polarity"]
        target_tokens = word_tokenize(text[0])  # NOTE changed from t.split() by pmhalvor
        label = "-exp-{0}".format(polarity)

        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]


def get_bio_holder(opinion):
    try:
        text, idxs = opinion["Source"]
    # will throw exception if the opinion target is None type
    except TypeError:
        return []
    except ValueError:
        return []
    # get the beginning and ending indices
    if len(text) > 1:
        updates = []
        #
        for t, idx in zip(text, idxs):
            bidx, eidx = idx.split(":")
            bidx = int(bidx)
            eidx = int(eidx)
            target_tokens = word_tokenize(t)  # NOTE changed from t.split() by pmhalvor
            label = "-holder"
            #
            tags = []
            for i, token in enumerate(target_tokens):
                if i == 0:
                    tags.append("B" + label)
                else:
                    tags.append("I" + label)
            updates.append((bidx, tags))
        return updates
    elif not idxs:  # NOTE elif clause added by pmhalvor
        return [(None, None)]
    else:
        bidx, eidx = idxs[0].split(":")
        bidx = int(bidx)
        eidx = int(eidx)
        target_tokens = word_tokenize(text[0])  # NOTE changed from text[0].split() by pmhalvor
        label = "-holder"
        #
        tags = []
        for i, token in enumerate(target_tokens):
            if i == 0:
                tags.append("B" + label)
            else:
                tags.append("I" + label)
        return [(bidx, tags)]
