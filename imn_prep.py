"""
This file should preprocess the NoReC_fine data to format IMN expects.

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

import json
import os
import shutil
# import nltk  # TODO only run first time through 
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm 
from norec_fine import get_bio_target
from norec_fine import get_bio_expression

#########  config  ###########
LOWER = True
OUTPUT_DIR = "../data/norec_fine/"
##############################

# read in json data
def read_data(filename):

    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


# extract targets, expressions, polarities, and sentences
def parse_data(data):
    """
    Returns lists of targets, expressions, target polarities, and sentences in IMN format.
    """
    targets = []
    expressions = []
    target_polarities = []
    sentences = []

    for i, line in tqdm(enumerate(data)):

        text = line["text"]
        tokens = word_tokenize(text)
        
        if line['opinions']:

            for opinion in line['opinions']:
                # encode target
                target = encode_target(text, tokens, opinion)

                # encode expression
                expression = encode_expression(text, tokens, opinion)

                # encode polarity
                target_polarity = encode_target_polarity(target, opinion)

                # tokenized sentence back as string
                sentence = ' '.join(tokens)
        else:
            # No opinion found
            target = [str(0) for _ in tokens]
            expression = [str(0) for _ in tokens]
            target_polarity = [str(0) for _ in tokens]
            sentence = ' '.join(tokens)
            
        targets.append(target)
        expressions.append(expression)
        target_polarities.append(target_polarity)
        sentences.append(sentence)
        

    return [targets, expressions, target_polarities, sentences]


def encode_target(text, tokens, opinion):
    """
    Encode labelled targets to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in orginal text is being labelled.
    """
    bio_target = get_bio_target(opinion)

    encoded = [str(0) for _ in tokens]

    if bio_target[0][0] is not None:

        for ele in bio_target:
            start_index = ele[0]
            bio_labels = ele[1]

            # Make sure correct index of token is labelled as target
            tokens_before = len(text[:start_index].split())
            encoded[tokens_before] = str(1)
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = str(2)
        
    return encoded


def encode_expression(text, tokens, opinion):
    """
    Encode labelled polar expressions to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in orginal text is being labelled.
    """
    bio_expression = get_bio_expression(opinion)

    encoded = [str(0) for _ in tokens]

    if bio_expression[0][0] is not None:

        for ele in bio_expression:
            start_index = ele[0]
            bio_labels = ele[1]

            tokens_before = len(text[:start_index].split())
            encoded[tokens_before] = str(1)
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = str(2)
        
    return encoded


def encode_target_polarity(target, opinion):
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

    encoded = [polar if int(token) > 0 else str(0) for token in target]

    return encoded


def store_data(filename, data):
    """
    Stores lists of data to file at filename.
    """

    if filename.split('.txt') == 1:
        filename += '.txt'
    
    with open(filename, 'w+') as f:
        for line in data:
            if "sentence" not in filename:
                f.write(" ".join(line))
            else:
                f.write(line)
            f.write("\n")


def run():
    datasets = ["train", "test", "dev"]

    try:
        os.mkdir(OUTPUT_DIR)
    except FileExistsError:
        answer = input(f"Directory {OUTPUT_DIR} already exists.\n Replace? [y/n]")
        if "n" in answer:
            return None
        shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    for dataset in datasets:
        data = read_data(f'../norec_fine/{dataset}.json')

        print(f"Parsing {dataset}...")
        parsed_package = parse_data(data)
        partitions = ["target", "opinion", "target_polarity", "sentence"]



        print(f"Storing {dataset}...")
        os.mkdir(OUTPUT_DIR+dataset)
        for partition, list_data in zip(partitions, parsed_package):
            store_data(OUTPUT_DIR+os.path.join(dataset, partition), list_data)


if __name__ == "__main__":
    run()
    

    