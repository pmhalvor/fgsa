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
from nltk.tokenize import wordpunct_tokenize


# convert json format to imn preprocessed format

def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data


# def parse_data(data):
#     ids = []
#     opinions = []
#     sentences = []
#     targets = []
#     target_polarities = []

#     for i, line in enumerate(data):
#         print('Line {}'.format(i))
#         for meta in line.keys():
#             if meta == 'opinions':
#                 for opinion in line[meta]:
#                     opinions.append(opinion)
#                     # for key in line[meta][opinion]:
#                     #     if key == 'target':
#                     #         targets.append = tokenize_target()
#                     #     print(key)
#                     targets.append(tokenize_target())
#                     sentences.append(line['text'])
#                     ids.append(line['sent_id'])
#                     target_polarities.append(opinion[])


def tokenize_sentence(sentence):
    tokens_long = wordpunct_tokenize(sentence)
    tokens = []

    prev = ""
    for token in tokens_long:
        if prev == "'":
            tokens.append(f"'{token}")
        elif token != "'":
            tokens.append(token)
        else:
            pass 
        prev = token        

    return tokens 

print(tokenize_sentence("a slightly more complex string, w/ punct's and (pizzazz)"))



# def tokenize_target(target, sentence):
#     sent_list = sentence.split(' ')
#     indexes = []
#     target_word = target[0]
#     target_indx = target[1]
#     for token in sent_list:
#         if token in target_word: 
#             indexes.append(1)
#         else:
#             indexes.append(0)

# def get_target_indexes(target, sentence):
#     target_word = target[0]
    
# def get_opinion_indexes(opinion, sentence):
#     opinion_list = sentence.split(' ')

# def split_sentence(sentence):
#     sent_list = []
#     for word in sentence.split(' '):
#         for p in string.punctuation:
#             if p in string.punctuation:
                
#         # if "'s" in word:
#         #     root = word.split("'s")[0]
#         #     sent_list.append(root)
#         #     sent_list.append("'s")
#         # # elif "-" == word:
#         # #     sent_list.append(word)
#         # else:
#         #     sent_list.append(word)
#     print(sent_list)



# sample = read_data('norec_fine/test.json')[:10]
# package = parse_data(sample)

'''
 {'sent_id': '201344-04-04',
  'text': 'Her har fokuset på funksjoner og indre kvaliteter , for første gang blitt kombinert med en ambisiøs satsning på et ekslusivt ytre .',
  'opinions': [{'Source': [[], []],
    'Target': [[], []],
    'Polar_expression': [['ambisiøs satsning på et ekslusivt ytre'],
     ['91:129']],
    'Polarity': 'Positive',
    'Intensity': 'Strong',
    'NOT': False,
    'Source_is_author': True,
    'Target_is_general': True,
    'Type': 'E'},
   {'Source': [[], []],
    'Target': [['satsning'], ['100:108']],
    'Polar_expression': [['ambisiøs'], ['91:99']],
    'Polarity': 'Positive',
    'Intensity': 'Standard',
    'NOT': False,
    'Source_is_author': True,
    'Target_is_general': False,
    'Type': 'E'},
   {'Source': [[], []],
    'Target': [['ytre'], ['125:129']],
    'Polar_expression': [['ekslusivt'], ['115:124']],
    'Polarity': 'Positive',
    'Intensity': 'Strong',
    'NOT': False,
    'Source_is_author': True,
    'Target_is_general': False,
    'Type': 'E'},
   {'Source': [[], []],
    'Target': [[], []],
    'Polar_expression': [['fokuset på funksjoner og indre kvaliteter'],
     ['8:49']],
    'Polarity': 'Positive',
    'Intensity': 'Strong',
    'NOT': False,
    'Source_is_author': True,
    'Target_is_general': True,
    'Type': 'E'},
   {'Source': [[], []],
    'Target': [[], []],
    'Polar_expression': [['fokuset på funksjoner og indre kvaliteter , for første gang blitt kombinert med en ambisiøs satsning på et ekslusivt ytre'],
     ['8:129']],
    'Polarity': 'Positive',
    'Intensity': 'Strong',
    'NOT': False,
    'Source_is_author': True,
    'Target_is_general': True,
    'Type': 'E'}]},
'''


# if __name__=='__main__':
#     user_input = input("Path to file: ")
#     read_data(user_input)
    