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
import string 
# import nltk  # TODO only run first time through 
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from norec_fine import get_bio_target
from norec_fine import get_bio_expression

#########  config  ###########
LOWER = True
##############################

# read in json data
def read_data(filename):

    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


# extract targets, polar expressions, polarity, full sentence (lower)
def parse_data(data):
    ids = []
    opinions = []
    sentences = []
    targets = []
    target_polarities = []

    for i, line in enumerate(data):

        print('Line {}'.format(i))

        text = line["text"]
        tokens = word_tokenize(text)
        
        print(text)

        if line['opinions']:

            for opinion in line['opinions']:
                # encode target
                target = encode_target(text, tokens, opinion)
                print("Target: \t", target)

                # encode expression
                expression = encode_expression(text, tokens, opinion)
                print("Expression: \t", expression)

                # encode polarity
                polarity = encode_target_polarity(text, tokens, opinion)
                print("Polarity: \t", polarity)

                # tokenized sentence back as string
                sentence = ' '.join(tokens)
                print("Sentence: \t", sentence)
        else:
            # No opinion found
            print(NotImplemented)
            
        print("\n\n\n")


def encode_target(text, tokens, opinion):
    """
    Encode labelled targets to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in orginal text is being labelled.
    """
    bio_target = get_bio_target(opinion)

    encoded = [0 for _ in tokens]

    if bio_target[0][0] is not None:

        for ele in bio_target:
            start_index = ele[0]
            bio_labels = ele[1]

            # Make sure correct index of token is labelled as target
            tokens_before = len(text[:start_index].split())
            encoded[tokens_before] = 1
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = 2
        
    return encoded


def encode_expression(text, tokens, opinion):
    """
    Encode labelled polar expressions to BIO, where B=1, I=2, O=0.
    Ensure the correct tokens in orginal text is being labelled.
    """
    bio_expression = get_bio_expression(opinion)

    encoded = [0 for _ in tokens]

    if bio_expression[0][0] is not None:

        for ele in bio_expression:
            start_index = ele[0]
            bio_labels = ele[1]

            tokens_before = len(text[:start_index].split())
            encoded[tokens_before] = 1
            for i in range(tokens_before + 1, tokens_before + len(bio_labels)):  # + 1 bc B is labelled above
                encoded[i] = 2
        
    return encoded


def encode_target_polarity(text, tokens, opinion):
    """
    Encode target polarities where:
        1 = positive
        2 = negative
        3 = neutral
        4 = confusing

    In norec_fine, no confusing label was given. In IMN, these are not evaluated against.
    Therefore, these will be exempt from labels.

    TODO: Tweak IMN to classify polar intensity, to help disambiguate confusing labels.

    """
    encoded = [0 for _ in tokens]


if __name__ == "__main__":
    sample = read_data('../norec_fine/test.json')[:2]
    package = parse_data(sample)

