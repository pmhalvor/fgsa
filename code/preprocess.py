"""
This file should preprocess the NoReC_fine data to format IMN expects.
Expects git@github.com:ltgoslo/norec_fine is cloned to parent directory '../'

IMN format:

norec/
    train/
        opinion.txt:
            0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

        sentence.txt:
            but the staff was so horrible to us
            to be completely fair , the only redeeming factor was the food , which was above average , but could n't make up for all the other deficiencies of teodora
            the food is uniformly exceptional , with a very capable kitchen which will proudly whip up whatever you feel like eating , whether it 's on the menu or not

        target.txt:
            0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0

        target_polarity.txt
            0 0 2 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0

    test/ 

"""

import argparse
import logging
import json
import os
import shutil

from nltk  import download 
from tqdm import tqdm 

from bio import get_bio_expression
from bio import get_bio_holder
from bio import get_bio_target
from bio import word_tokenize
import config

#########  config  ###########
config.log_template(job="pre", name="adding-BertTokenizer")
ERROR_COUNT = 0
KNOWN_ERRONEOUS_IDS = ['703281-03-01', '705034-09-03']
LOWER = True
OUTPUT_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"

try:
    word_tokenize('text')
except LookupError:
    nltk.download('punkt')  # only needed for first run
##############################

# read in json data
def read_data(filename):

    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


# extract targets, expressions, polarities, and sentences
def parse_data(data, interactive=False):
    """
    Returns Dict[(str, List)] of targets, expressions, target polarities, and sentences in IMN format.
    """
    expressions = []
    holders = []
    sentences = []
    targets = []
    target_polarities = []

    data_itr = tqdm(enumerate(data)) if interactive else enumerate(data)
    for i, line in data_itr:

        if line["sent_id"] in KNOWN_ERRONEOUS_IDS:
            continue

        text = line["text"]
        tokens = word_tokenize(text)  # NOTE potential flaw. Cross check with res/res_15

        expression = [str(0) for _ in tokens]
        holder = [str(0) for _ in tokens]
        target = [str(0) for _ in tokens]
        target_polarity = [str(0) for _ in tokens]
        
        if line['opinions']:

            for opinion in line['opinions']:
                # encode target
                target = encode_target(text, tokens, opinion, target)

                # encode expression
                expression = encode_expression(text, tokens, opinion, expression)

                # encode holder
                # TODO Implement holder
                holder = encode_holder(text, tokens, opinion, holder)

                # encode polarity
                target_polarity = encode_target_polarity(target, opinion, target_polarity)

        # tokenized sentence back as string
        sentence = ' '.join(tokens).lower()

        expressions.append(expression)
        holders.append(holder)
        sentences.append(sentence)
        targets.append(target)
        target_polarities.append(target_polarity)
        
    return {
        'opinion': expressions,  # NOTE expression -> opinion due IMN format expectations
        'holder': holders, 
        'sentence': sentences,
        'target': targets, 
        'target_polarity': target_polarities, 
    }


def encode_target(text, tokens, opinion, target):
    """
    Encode labelled targets to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in original text is being labelled.
    """
    bio_target = get_bio_target(opinion)


    encoded = [str(0) for _ in tokens]  # NOTE FIXME

    if bio_target[0][0] is not None:

        for ele in bio_target:
            start_index = ele[0]
            bio_labels = ele[1]

            # Make sure correct index of token is labelled as target
            tokens_before = len(word_tokenize(text[:start_index]))  # tokenized words up to target start
            encoded[tokens_before] = str(1)  # token count before == start index of target bc start at 0  
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = str(2)


    # check encoded target matches dataset target
    check(encoded, tokens, opinion, 'Target')

    # fill target with new encoding
    encoded_target = [
        enc if int(enc) > 0 and int(tar) == 0 else tar 
        for enc, tar in zip(encoded, target)
    ]
        
    return encoded_target


def encode_expression(text, tokens, opinion, expression):
    """
    Encode labelled polar expressions to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in original text is being labelled.
    """
    bio_expression = get_bio_expression(opinion)

    encoded = [str(0) for _ in tokens]

    if bio_expression[0][0] is not None:

        for ele in bio_expression:
            start_index = ele[0]
            bio_labels = ele[1]

            tokens_before = len(word_tokenize(text[:start_index]))
            try:
                encoded[tokens_before] = str(1)
            except IndexError:
                print(tokens_before)
                print(text)
                print(start_index)
                print(text[:start_index])
                print(tokens)
                print(bio_expression)
                print(opinion)
                quit()
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = str(2)
        
    # check encoded expression matches dataset expression
    check(encoded, tokens, opinion, 'Polar_expression')

    # fill expression with new encoding
    encoded_expression = [
        enc if int(enc) > 0 and int(tar) == 0 else tar 
        for enc, tar in zip(encoded, expression)
    ]
        
    return encoded_expression


def encode_holder(text, tokens, opinion, holder):
    """
    Encode labelled polar expressions to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in original text is being labelled.
    """
    bio_holder = get_bio_holder(opinion)
    # TODO check this is all i need
    encoded = [str(0) for _ in tokens]

    if bio_holder[0][0] is not None:

        for ele in bio_holder:
            start_index = ele[0]
            bio_labels = ele[1]

            tokens_before = len(word_tokenize(text[:start_index]))
            encoded[tokens_before] = str(1)
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = str(2)
        
    # check encoded holder matches dataset holder
    check(encoded, tokens, opinion, 'Source')

    # fill holder with new encoding
    encoded_holder = [
        enc if int(enc) > 0 and int(tar) == 0 else tar 
        for enc, tar in zip(encoded, holder)
    ]
    
    return encoded_holder


def encode_target_polarity(target, opinion, target_polarity):
    """
    Encode target polarities where:
        1 = positive
        2 = negative
        3 = neutral
        4 = confusing

    In norec_fine, no "confusing" or "neutral" labels were given. 
    In IMN, "confusing" was not evaluated against, but "neutral" was.
    For simplicity, these will both be exempt from labeling here.

    TODO: Tweak IMN to classify polar intensity and/or fact-implied non-personal labels.
    TODO: Use Neutral for fact implied non-personal
    """
    polarity = opinion["Polarity"]
    if polarity:
        if "Positive" in polarity:
            polar = str(1)
        elif "Negative" in polarity:
            polar = str(2)
    else:
        polar = str(3)

    encoded = target_polarity

    for i, (token, polarity) in enumerate(zip(target, target_polarity)):
        if int(polarity) == 0:
            if int(token) > 0:
                encoded[i] = polar

    return encoded


def check(encoded, tokens, opinion, attribute):
    # check encoded attribute matches dataset attribute
    if len(opinion[attribute][0]) > 0:
        # double join to match tokenized formatting
        expected = ' '.join(word_tokenize(' '.join(opinion[attribute][0]))).lower()
    else:
        expected = ''        

    encoded_tokens = []
    for i, enc in enumerate(encoded):
        if int(enc) > 0: 
            encoded_tokens.append(tokens[i])
    encoded = ' '.join(encoded_tokens).lower()

    
    global ERROR_COUNT
    try:
        assert expected == encoded
    except AssertionError:
        if len(expected) == len(encoded):
            # most likely swapped orders. double check tokens match 
            # TODO check opposite way as well?
            for ele in expected.split():
              if ele not in encoded:
                logging.info(f'Conflicting {attribute}:')
                logging.info(expected)
                logging.info(encoded)
                ERROR_COUNT += 1

        else:
            logging.info(f'Conflicting {attribute}:')
            logging.info(expected)
            logging.info(encoded)
            ERROR_COUNT += 1


def store_data(filename, data):
    """
    Stores lists of data to file at filename.
    """

    if len(filename.split('.txt')) == 1:
        filename += '.txt'
    
    with open(filename, 'w+') as f:
        for line in data:
            if "sentence" not in filename:
                f.write(" ".join(line))
            else:
                f.write(line)
            f.write("\n")
    print(f"Saved {filename}")


def run(interactive=True, overwrite=False):
    datasets = [
        "test", 
        "train", 
        "dev",
    ]
    print("interactive: {}".format(interactive))
    print("overwrite: {}".format(overwrite))

    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        if interactive:
            # allow user to decide whether or not to overwrite
            answer = input(f"Directory {OUTPUT_DIR} already exists.\n Replace? [y/n]")
            if "n" in answer:
                pass
            shutil.rmtree(OUTPUT_DIR)
            os.mkdir(OUTPUT_DIR)
        elif overwrite:
            # force overwrite
            shutil.rmtree(OUTPUT_DIR)
            os.mkdir(OUTPUT_DIR)
        else:
            return None

    except FileNotFoundError as e:
        print("Directory not found. Try using a full path for OUTPUT_DIR in config.")
        return None

    for dataset in datasets:
        data = read_data(f'../norec_fine/{dataset}.json')

        print(f"\n Parsing {dataset}...")
        logging.info(f"\n Parsing {dataset}...")
        parsed_package = parse_data(data, interactive=interactive)
        partitions = [
            'opinion',
            'holder',
            'sentence',
            'target',
            'target_polarity',
        ]

        global ERROR_COUNT
        logging.info(f'Errors for {dataset}: {ERROR_COUNT}')
        logging.info('- - - - - - - - - - - - - - - - - - - - - - - - - - -')
        ERROR_COUNT = 0

        print(f"Storing {dataset}...")
        os.mkdir(OUTPUT_DIR+dataset)
        for partition in partitions:
            filepath = os.path.join(OUTPUT_DIR, dataset, partition)
            store_data(filepath, parsed_package[partition])


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", dest="interactive", type=int, default=0, help="Interactive if overwrite necessary")
    parser.add_argument("-o", "--overwrite", dest="overwrite", type=int, default=1, help="Force overwrite (if not interactive)")
    args = parser.parse_args()

    run(**vars(args))

    print("complete")
    

    