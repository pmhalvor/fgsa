# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.baseline_studies import EMBEDDINGvsTRAIN_EMBEDDINGS
    from exam.utils.baseline_studies import NUM_LAYERSvsHIDDEN_DIM
    from exam.utils.baseline_studies import LEARNING_RATEvsDROPOUT
    from exam.utils.baseline_studies import BATCH_SIZEvsEPOCHS
except:
    from utils.baseline_studies import EMBEDDINGvsTRAIN_EMBEDDINGS
    from utils.baseline_studies import NUM_LAYERSvsHIDDEN_DIM
    from utils.baseline_studies import LEARNING_RATEvsDROPOUT
    from utils.baseline_studies import BATCH_SIZEvsEPOCHS

import torch


non_pos_tagged_corporas = {
    '58': 100,
    '77': 100,
    '79': 100,
    '81': 100,
    '84': 100,
    '86': 100,
    '88': 100,
    '90': 100,
    '92': 100,
    '94': 100,
    '96': 100,
    '98': 100,
    '100': 100,
    '102': 100,
    '104': 100,
    '106': 100,
    '108': 100,
    '110': 100,
    '112': 100,
    '114': 100,
    '116': 100,
    '118': 100,
    '120': 100,
    '122': 100,
    '124': 100,
    '126': 100,
    '127': 50,
    '128': 300,
    '129': 600
}

pos_tagged_corporas = {
    '76': 100,
    '78': 100,
    '80': 100,
    '83': 100,
    '85': 100,
    '87': 100,
    '89': 100,
    '91': 100,
    '93': 100,
    '95': 100,
    '97': 100,
    '99': 100,
    '101': 100,
    '103': 100,
    '105': 100,
    '107': 100,
    '109': 100,
    '111': 100,
    '113': 100,
    '115': 100,
    '117': 100,
    '119': 100,
    '121': 100,
    '123': 100,
    '125': 100,
    '130': 500,
    '131': 300,
    '132': 600,
    '133': 50,
    '134': 300,
    '135': 600,
    '189': 300
}

bert_corporas = {'216': 768}

elmo_corporas = {'217': 2048, '218': 2048}


# dict with current best params (updated after every study)
params = {
    'TRAIN_DATA': "data/train.conll",
    'DEV_DATA': "data/dev.conll",
    'TEST_DATA': "data/test.conll",
    'verbose': True,
    'random_state': 1,
    'BATCH_SIZE': 32,
    'HIDDEN_DIM': 50,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'output_dim': 5,
    'NUM_LAYERS': 1,
    'DROPOUT': 0.1,
    'LEARNING_RATE': 0.01,
    'TRAIN_EMBEDDINGS': True,
    'EPOCHS': 20,
    # 'EMBEDDINGS_DIR': "exam/saga/",
    'EMBEDDINGS_DIR': "/cluster/shared/nlpl/data/vectors/latest/",
    'EMBEDDINGS': "58"
}

# parameter space for each study
space = {
    'EMBEDDINGvsTRAIN_EMBEDDINGS': {
        'par_1': list(non_pos_tagged_corporas.keys()),
        'par_2': [True, False],
    },
    'NUM_LAYERSvsHIDDEN_DIM': {
        'par_1': [1, 3, 5],
        'par_2': [5, 10, 50, 100, 500],
    },
    'LEARNING_RATEvsDROPOUT': {
        'par_1': [0.1, 0.01, 0.001, 0.0001],
        'par_2': [0.1, 0.2, 0.3],
    },
    'BATCH_SIZEvsEPOCHS': {
        'par_1': [25, 32, 36, 40, 44, 50],
        'par_2': [5, 10, 25, 50],
    }
}

################# 1st study
print('First study #=#=#=#=#')
params.pop('EMBEDDINGS')
params.pop('TRAIN_EMBEDDINGS')
params['EMBEDDINGS'], params['TRAIN_EMBEDDINGS'] = \
    EMBEDDINGvsTRAIN_EMBEDDINGS(
        par_1=space['EMBEDDINGvsTRAIN_EMBEDDINGS']['par_1'],
        par_2=space['EMBEDDINGvsTRAIN_EMBEDDINGS']['par_2'],
        out_path_filename="outputs/baseline_EMBEDDINGvsTRAIN_EMBEDDINGS",
        **params
    ).run()._best_params()
#####################################

# # ################# 2nd study
# print('Second study #=#=#=#=#')
# params.pop('NUM_LAYERS')
# params.pop('HIDDEN_DIM')
# params['NUM_LAYERS'], params['HIDDEN_DIM'] = NUM_LAYERSvsHIDDEN_DIM(
#     par_1=space['NUM_LAYERSvsHIDDEN_DIM']['par_1'],
#     par_2=space['NUM_LAYERSvsHIDDEN_DIM']['par_2'],
#     out_path_filename="outputs/baseline_NUM_LAYERSvsHIDDEN_DIM",
#     **params
# ).run()._best_params()
# #####################################


# print('Third study #=#=#=#=#')
# # ################# 3rd study
# params.pop('LEARNING_RATE')
# params.pop('DROPOUT')
# params['LEARNING_RATE'], params['DROPOUT'] = LEARNING_RATEvsDROPOUT(
#     par_1=space['LEARNING_RATEvsDROPOUT']['par_1'],
#     par_2=space['LEARNING_RATEvsDROPOUT']['par_2'],
#     out_path_filename="outputs/baseline_LEARNING_RATEvsDROPOUT",
#     **params
# ).run()._best_params()
# # #####################################

print('Fourth study #=#=#=#=#')
# ################# 4th study
params.pop('BATCH_SIZE')
params.pop('EPOCHS')
params['BATCH_SIZE'], params['EPOCHS'] = BATCH_SIZEvsEPOCHS(
    par_1=space['BATCH_SIZEvsEPOCHS']['par_1'],
    par_2=space['BATCH_SIZEvsEPOCHS']['par_2'],
    out_path_filename="outputs/baseline_BATCH_SIZEvsEPOCHS",
    **params
).run()._best_params()
# #####################################
