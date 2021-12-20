"""
Author: pmhalvor
Title: IMN/model
Description: IMN architecture written in PyTorch with Python 3.6
"""

import logging 
import torch.nn
import torch 
import numpy as np
from torch.nn.modules.dropout import Dropout

from my_layers import Conv1DWithMasking
from my_layers import Self_attention


############### logging ###############
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
#######################################


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
        shared_cnn_components = []
        for i in range(args.shared_layers):
            print('Shared CNN layer %s'%i)
            shared_cnn_components.append(torch.nn.Dropout(args.dropout_prob))

            if i == 0:
                self.shared_conv_0_1 = Conv1DWithMasking(
                    in_channels = 1,  # TODO check this makes sense
                    # conv dim
                    out_channels = args.cnn_dim/2,
                    # filter size
                    kernel_size = 3,
                    # other params
                    padding_mode = 'reflect',
                )
                self.shared_conv_0_2 = Conv1DWithMasking(
                    in_channels = 1,  # TODO check this makes sense
                    # conv dim
                    out_channels = args.cnn_dim/2,
                    # filter size
                    kernel_size = 5,
                    # other params
                    padding_mode = 'reflect',
                )

                # these need to be relu activated 
                # then concatenated in self.forward()                

            else:
                shared_cnn_components.append(
                    Conv1DWithMasking(
                        # input channels should be 1, right?
                        in_channels = 1,  # TODO check this makes sense
                        # conv dim
                        out_channels = args.cnn_dim,
                        # filter size
                        kernel_size = 5,
                        # other params
                        padding_mode = 'reflect',
                    )
                )
        
        self.shared_cnn = torch.nn.Sequential(*shared_cnn_components)
        
        #####################################
        # Task-specific layers
        #####################################

        #### AE: Aspect extraction task ####
        aspect_cnn_list = []
        for a in range(args.aspect_layers):
            print('Aspect extraction layer %s'%a)
            aspect_cnn_list.append(torch.nn.Dropout(args.dropout_prob))
            aspect_cnn_list.append(Conv1DWithMasking(
                # input channels should be 1, right?
                in_channels = 1,  # TODO check this makes sense
                # conv dim
                out_channels = args.cnn_dim,
                # filter size
                kernel_size = 5,
                # other params
                padding_mode = 'reflect',
            ))
            aspect_cnn_list.append(torch.nn.ReLU())

        # combine all layers to single object
        self.aspect_cnn = torch.nn.Sequential(*aspect_cnn_list)

        # feed through fully connected dense layer w/ softmax activation
        # TODO check output size in Linear in line below here
        self.aspect_dense = torch.nn.Sequential(
            torch.nn.Linear(args.cnn_dim, args.cnn_dim),
            torch.nn.Softmax()
        )



        #### AS: Aspect sentiment task ####
        sentiment_cnn_list = [] 
        for b in range(args.senti_layers):
            print('Sentiment classification layer %s'%b)
            sentiment_cnn_list.append(torch.nn.Dropout(args.dropout_prob))
            sentiment_cnn_list.append(
                Conv1DWithMasking(
                    # input channels should be 1, right?
                    in_channels = 1,  # TODO check this makes sense
                    # conv dim
                    out_channels = args.cnn_dim,
                    # filter size
                    kernel_size = 5,
                    # other params
                    padding_mode = 'reflect',

                )
            )

        # combine all layers to single object
        self.sentiment_cnn = torch.nn.Sequential(*sentiment_cnn_list)

        # attention layer
        # TODO swap this out for my_layers.Self_attention()
        # needs to be trainable on gold opinions (see line 233 in OG)
        self.sentiment_att = torch.nn.MultiheadAttention(
                embed_dim=args.emb_dim,
                num_heads=1,
            )

        # dense layer for probability 
        self.sentiment_dense = torch.nn.Sequential(
            torch.nn.Linear(args.cnn_dim, args.cnn_dim),
            torch.nn.Softmax()
        )



        if args.use_doc:
            # fill out for document level classifications similar to:
            """
            # DS specific layers
            doc_senti_cnn = Sequential()
            for c in xrange(args.doc_senti_layers):
                print 'Document-level sentiment layers %s'%c
                doc_senti_cnn.add(Dropout(args.dropout_prob))
                doc_senti_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                        activation='relu', padding='same', kernel_initializer=my_init, name='doc_sentiment_cnn_%s'%c))

            doc_senti_att = Attention(name='doc_senti_att')
            doc_senti_dense = Dense(3, name='doc_senti_dense')
            # The reason not to use the default softmax is that it reports errors when input_dims=2 due to 
            # compatibility issues between the tf and keras versions used.
            softmax = Lambda(lambda x: K.tf.nn.softmax(x), name='doc_senti_softmax')

            # DD specific layers
            doc_domain_cnn = Sequential()
            for d in xrange(args.doc_domain_layers):
                print 'Document-level domain layers %s'%d 
                doc_domain_cnn.add(Dropout(args.dropout_prob))
                doc_domain_cnn.add(Conv1DWithMasking(filters=args.cnn_dim, kernel_size=5, \
                        activation='relu', padding='same', kernel_initializer=my_init, name='doc_domain_cnn_%s'%d))

            doc_domain_att = Attention(name='doc_domain_att')
            doc_domain_dense = Dense(1, activation='sigmoid', name='doc_domain_dense')

            """
            pass

        # re-encoding layer
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(args.cnn_dim, args.cnn_dim),
            torch.nn.ReLU()
        )


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
            # doc_output_1 = self.word_emb(doc_input_1)
            # doc_output_2 = self.word_emb(doc_input_2)
            # if self.args.use_domain_emb:
            #     # mask out the domain embeddings
            #     doc_output_2 = '' # Remove_domain_emb()(doc_output_2)
            pass


        ##########################################
        # Shared features
        ##########################################
        sentence_output_1 = self.shared_conv_0_1(sentence_output)
        sentence_output_2 = self.shared_conv_0_2(sentence_output)
        sentence_output = torch.cat((sentence_output_1, sentence_output_2))


        ### start here
        ## stopping bc concatenated (word embeddings, sentence_output)
        ## in build makes Sequential not an option. 
        


        ##########################################
        # aspect-level message passing operation
        ##########################################
        aspect_output = sentence_output
        sentiment_output = sentence_output

        for i in range(self.args.interactions+1):
            print('Interaction number %s'%i)

            ###  AE  ###
            if self.args.aspect_layers > 0:
                aspect_output = self.aspect_cnn(aspect_output)
            aspect_output = torch.cat((word_embeddings, aspect_output))
            aspect_output = torch.nn.Dropout(self.args.dropout_prob)(aspect_output)
            aspect_probs  = self.aspect_dense(aspect_output)

            ###  AS  ###
            if self.args.senti_layers > 0:
                sentiment_output = self.sentiment_cnn(sentiment_output)

            # should have some way of including gold opinions
            sentiment_output = self.sentiment_att(sentiment_output)
            sentiment_output = torch.cat((init))


        

        




class IMN_BERT(torch.nn.module):
    def __init__(self) -> None:
        super().__init__()