"""
Author: pmhalvor
Title: IMN/model
Description: IMN architecture written in PyTorch with Python 3.6
"""

import logging 
import torch 
import numpy as np

from my_layers import Conv1DWithMasking


############### logging ###############
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
#######################################


# Custom CNN kernel initializer
# Use the initialization from Kim et al. (2014) for CNN kernel
def my_init(shape):
    return 0.01 * np.random.standard_normal(size=shape)


def create_model(args, vocab, nb_class, overall_maxlen, doc_maxlen_1, doc_maxlen_2):

    # Function that initializes word embeddings
    def init_emb(emb_matrix, vocab, emb_file_gen, emb_file_domain):
        """
        Function that initializes word embeddings.
        Loads from `emb_file_gen` (and `emb_file_domain` if activated).

        NOTE: Datasets used for IMN experiemnt differ from NOREC, 
        so adaptations will be needed when running as baseline.

        Parameters:
            emb_matrix: empty word look-up matrix to be filled with loaded embeddings of size [n, 300] (or [n, 400] if domain specific embeddings are activated)

            vocab: not exactly sure yet, come back (TODO)

            emb_file_gen: pretrained generalized embeddings of size [n, 301]

            emb_file_domain: pretrained domain specific embeddings

        Return:
            emb_matrix: same size as before, but now updated with pretrained values
        """

        print('Loading pretrained general word embeddings and domain word embeddings ...')

        counter_gen = 0.
        # NOTE: emebddings data may be stored differently (check later)
        pretrained_emb = open(emb_file_gen)
        for line in pretrained_emb:
            tokens = line.split()

            if len(tokens) != 301:
                continue

            word = tokens[0]
            vec = tokens[1:]

            try: 
                emb_matrix[0][vocab[word]][:300] = vec
                counter_gen += 1
            except KeyError:
                pass

        if args.use_domain_emb:
            counter_domain = 0.
            pretrained_emb = open(emb_file_domain)
            for line in pretrained_emb:
                tokens = line.split()

                if len(tokens) != 101:
                    continue

                word = tokens[0]
                vec[1:]

                try: 
                    emb_matrix[0][vocab[word]][300:] = vec
                    counter_domain += 1
                except KeyError:
                    pass

        pretrained_emb.close()
        logger.info('%i/%i word vectors initialized by general embeddings (hit rate: %.2f%%)' % (counter_gen, len(vocab), 100*counter_gen/len(vocab)))

        if args.use_domain_emb:
            logger.info('%i/%i word vectors initialized by domain embeddings (hit rate: %.2f%%)' % (counter_domain, len(vocab), 100*counter_domain/len(vocab)))

        return emb_matrix


    # Build model
    logger.info('Building model ...')
    print('Building model ... \n\n\n')

    vocab_size = len(vocab)

    ###########################################
    # Inputs
    ###########################################
    print('Input layer')

    # sequence of token indices for aspect-level data
    """
    At this point, my data will be served as a tensor created per batch 
    """
    sentence_input = ''



class IMN(torch.nn.module):
    def __init__(
        self, 
        args, 
        vocab, 
        nb_class, 
        overall_maxlen, 
        doc_maxlen_1, 
        doc_maxlen_2
    ) -> None:
        super().__init__()
        print('Initialize model ...')


        # store parameters as attributes in model
        self.args = args
        self.vocab = vocab
        self.nb_class = nb_class 
        self.overall_maxlen = overall_maxlen 
        self.doc_maxlen_1 = doc_maxlen_1 
        self.doc_maxlen_2 = doc_maxlen_2


        #####################################
        # Shared word embedding layer
        #####################################
        self.vocab_size = len(vocab)
        self.word_emb = torch.nn.Embedding(
            self.vocab_size,
            args.emb_dim,
            # mask_zero=True # think this is already True
        )

        # NOTE from here on out document level analysis is ignored (to be implemeneted later)
        shared_list = []
        for i in range(args.shared_layers):
            print('Shared CNN layer %s'%i)
            shared_list.append(torch.nn.Dropout(args.dropout_prob))

            if i == 0:
                conv_1 = Conv1DWithMasking(
                    in_channels = 1,  #TODO check this makes sense
                    out_channels = args.cnn_dim/2,
                    kernel_size = 3,
                    

                    # filter size
                    # conv dim
                    # other params
                )
                conv_2 = Conv1DWithMasking(
                    # filter size
                    # conv dim
                    # other params
                )
                shared_list.append(conv_1)
                shared_list.append(conv_2)
            else:
                conv_1 = Conv1DWithMasking(
                    # filter size
                    # conv dim
                    # other params
                )
                conv_2 = Conv1DWithMasking(
                    # filter size
                    # conv dim
                    # other params
                )
        
        





        self.shared_cnn = torch.nn.Sequential(**shared_list)


    def forward(
        self, sentence_input, 
        doc_input_1=None, 
        doc_input_2=None
    ):
        # aspect-level inputs
        word_embeddings = self.word_emb(sentence_input)
        sentence_output = word_embeddings

        # doc-level inputs
        if self.args.use_doc:
            doc_output_1 = self.word_emb(doc_input_1)
            doc_output_2 = self.word_emb(doc_input_2)
            if self.args.use_domain_emb:
                # mask out the domain embeddings
                doc_output_2 = '' # Remove_domain_emb()(doc_output_2)
        




class IMN_BERT(torch.nn.module):
    def __init__(self) -> None:
        super().__init__()