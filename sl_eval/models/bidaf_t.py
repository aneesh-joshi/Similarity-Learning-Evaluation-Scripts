"""This file conatains an implementation of the BiDAF-T model introduced in the QA-Transfer paper

You can read more here:
Question Answering through Transfer Learning from Large Fine-grained Supervision Data
https://arxiv.org/pdf/1702.02171.pdf

The above paper claims that pretraining the BiDirectional Attention (https://arxiv.org/abs/1611.01603) with the SQUAD dataset
for span supervision and then converting the model to a QA model by changing the last layer (making BiDAF -> BiDAF-T) and further
finetuning it leads to great results (0.79 MAP on WikiQA)

It also claims pretraining the BiDAF-T model on SQUAD-T (a QA version of SQUAD (read more in misc_scripts/squad2QA.py))
and fine tuning on WikiQA train gets 0.76 MAP on WikiQA test.
This script is an attempt to reproduce this claim.

Beware: This doesn't implement any pretraining on span level QA at. My though process is "if I can reproduce the result with just SQUAD-T dataset,
that will itself be SOTA. If it works, there might be a point in investing in Span level QA." So, far the result wasn't reprocuded. :(


Notes
------
It's important to understand that the BiDAF model takes inputs in the format
question = (batch_size, num_question_words)
passage = (batch_size, num_passage_sentences, num_passage_words)

Effectively, a passage is of the form:
question = [["who", "is", "there", PAD]]
(1, 4) -- (batch_size, num_question_words)

passage = [["Hello", "there", PAD, PAD, PAD],
           ["General", "Kenobi", "Sir", PAD, PAD],
           [PAD, PAD, PAD, PAD, PAD ]]

(1, 3, 5) -- (batch_size, num_passage_sentences, num_passage_words)

Notice that the words have been padded to length 5 and the last sentence is padded fully since the num_passage_words is
set to 3(say).

In the original implementaion, this is masked. However, I haven't masked it in this implemenatation and hope that the model will learn to
ignore it. There are 2 reasons I didn't mask it:
1. I currently don't know how to and it will take some time to learn it. Maybe a TODO
2. It will complicate the implementation which I have tried to keep extremely simple

"""


from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed,\
                         LSTM, Bidirectional, Lambda, Reshape, Activation, Masking, Conv1D
from keras.models import Model
from keras.utils.np_utils import to_categorical
import keras.backend as K

from .utils.custom_layers import Highway

import numpy as np
import tensorflow as tf
import random as rn
import os
from collections import Counter
import gensim.downloader as api
import logging
import hashlib
from numpy import random as np_random
import string
from keras import optimizers


# Set random seed for reproducible results.
# For more details, read the keras docs:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import os
os.environ['PYTHONHASHSEED'] = '0'
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


logger = logging.getLogger(__name__)

class BiDAF_T:
    def __init__(self, queries, docs, labels, kv_model, max_passage_words=100, max_passage_sents=1, max_question_words=40,
        char_embedding_dim=8, batch_size=50, unk_handle_method='zero', pad_handle_method='zero', optimizer='adam',
        n_epochs=5, n_encoder_hidden_nodes=200, max_word_charlen=25, depth=5, filters=100, word_embedding_dim=100,
        steps_per_epoch=1):

        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.kv_model = kv_model
        self.max_passage_words = max_passage_words
        self.max_passage_sents = max_passage_sents
        self.total_passage_words = self.max_passage_sents * self.max_passage_words
        self.max_question_words = self.max_passage_words
        self.word_embedding_dim = word_embedding_dim
        self.batch_size = batch_size
        self.unk_handle_method = unk_handle_method
        self.pad_handle_method = pad_handle_method
        self.char_embedding_dim = char_embedding_dim
        self.n_epochs = n_epochs
        self.max_word_charlen = max_word_charlen
        self.loss = 'categorical_crossentropy'
        self.n_encoder_hidden_nodes = n_encoder_hidden_nodes
        self.char_pad_index = 0
        self.pad_word_index = 0
        self.depth = depth
        self.filters = filters
        self.model = None
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # ada = optimizers.Adadelta(lr=0.5)
        # optimizer = ada
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = 'adam'

        self.char2index = {}
        self.word2index = {}

        self.build_vocab()
        self.train()

    def build_vocab(self):
    
        n_queries, n_docs = 0, 0
        q_lens, doc_lens, n_docs_per_query = [], [], []

        for i, char in enumerate(string.printable, start=1):
            self.char2index[char] = i

        # Create word indexes from training data
        word_counter = Counter()
        for q, docs, labels in zip(self.queries, self.docs, self.labels):
            n_queries += 1
            q_lens.append(len(q))
            n_docs_per_query.append(len(docs))
            word_counter.update(q)
            for d in docs:
                n_docs += 1
                word_counter.update(d)
                doc_lens.append(len(d))

        self.vocab_size = len(word_counter)
        logger.info('There are %d vocab words in the training data', self.vocab_size)
        logger.info('There are %d queries and %d docs in the training data', n_queries, n_docs)

        self.word2index = {}
        for i, word in enumerate(word_counter.keys(), start=1):
            self.word2index[word] = i

        self.pad_word_index = 0
        self.unk_word_index = self.vocab_size + 1

        logger.info(
            'The pad_word is set to the index: %d and the unkwnown words will be set to the index %d',
            self.pad_word_index, self.unk_word_index
        )

        num_embedding_words = 0
        num_non_embedding_words = 0

        for word in self.word2index:
            if word in self.kv_model:
                num_embedding_words += 1
            else:
                num_non_embedding_words += 1

        logger.info(
            "There are %d words in the embdding matrix and %d which aren't there",
            num_embedding_words, num_non_embedding_words
        )

        # 1 for pad word, 1 for unk_word in non-train data
        self.embedding_matrix = np.zeros((1 + len(self.kv_model.vocab) + 1 + num_embedding_words , self.kv_model.vector_size))

        if self.pad_handle_method == 'zero':
            self.embedding_matrix[self.pad_word_index] = np.zeros((self.kv_model.vector_size))
        elif self.pad_handle_method == 'random':
            self.embedding_matrix[self.pad_word_index] = np.random.random((self.kv_model.vector_size))
        else:
            raise ValueError('Unkown pad handle method')

        if self.unk_handle_method == 'zero':
            self.embedding_matrix[self.unk_word_index] = np.zeros((self.kv_model.vector_size))
        elif self.unk_handle_method == 'random':
            self.embedding_matrix[self.unk_word_index] = np.random.random((self.kv_model.vector_size))
        else:
            raise ValueError('Unkown unk handle method')

        for word in self.word2index:
            if word in self.kv_model:
                self.embedding_matrix[self.word2index[word]] = self.kv_model[word]
            else:
                self.embedding_matrix[self.word2index[word]] = self._seeded_vector(word, self.kv_model.vector_size)

        additional_embedding_index = self.pad_word_index + 1
        for word in self.kv_model.vocab:
            if word not in self.word2index:
                self.embedding_matrix[additional_embedding_index] = self.kv_model[word]
                self.word2index[word] = additional_embedding_index
                additional_embedding_index += 1

        del self.kv_model



    def _char_word(self, word):
        """Converts a str word into a list of its character indices with pads

        Parameters
        ----------
        word : str
        """
        list_word = list(word)
        indexed_word = [self.char2index[l] for l in list_word]
        indexed_word = indexed_word + (self.max_word_charlen - len(indexed_word))*[self.char_pad_index]
        if len(indexed_word) > self.max_word_charlen:
            logger.info('Word: %s is really long of length: %d. Clipping to length %d for now \n\n\n',
                         word, len(indexed_word), self.max_word_charlen)
            indexed_word = indexed_word[:self.max_word_charlen]
        return indexed_word


    def _make_sentence_indexed_padded(self, sentence, max_len):
        """Returns a list of ints for a given list of strings
        
        Parameters
        ----------
        sentence : list of str

        Usage
        -----
        >>>_make_sentence_indexed_padded('I am Aneesh and I like chicken'.split(), max_passage_words)
        [23794, 1601, 23794, 115, 23794, 764, 19922, 0, 0, 0, 0, 0]
        """

        assert type(sentence) == list
        str_sent = sentence
        sentence = [self.word2index[word] if word in self.word2index else self.unk_word_index for word in sentence]
        while len(sentence) < max_len:
            sentence.append(self.pad_word_index)
        if len(sentence) > max_len:
            logger.info("Max length of %d wasn't long enough for sentence of length %d. Truncating sentence for now",
                max_len, len(sentence))
            sentence = sentence[:max_len]
        return sentence

    def _make_sentence_indexed_padded_charred(self, sentence, max_len):
        """Returns a list of list of ints for a given list of strings
        Same as _make_sentence_indexed_padded except each word is a list of ints
        padded to max_word_charlen
                
        Parameters
        ----------
        sentence : list of str
        """
        assert type(sentence) == list
        
        str_sent = sentence
        sentence = [self._char_word(word) for word in sentence]
        while len(sentence) < max_len:
            sentence += [[self.char_pad_index] * self.max_word_charlen]
        if len(sentence) > max_len:
            logger.info("Max length of %d wasn't long enough for sentence:%s of length %d. Truncating sentence for now",
                max_len, str(str_sent), len(sentence))
            sentence = sentence[:max_len]
        return sentence

    def _get_full_batch_iter(self, pair_list, batch_size):
        """Returns batches with alternate positive and negative docs taken from
        the `pair_list`. Since each question has a positive and neagative counter part,
        we divide batch_size by 2 (batch_size // 2 * 2)
        """
        X1, X2, y = [], [], []
        cX1, cX2 = [], []
        batch_size = batch_size//2
        while True:
            for i, (query, pos_doc, neg_doc, cquery, cpos_doc, cneg_doc) in enumerate(pair_list):
                X1.append(query)
                cX1.append(cquery)

                X2.append(pos_doc)
                cX2.append(cpos_doc)

                y.append([0, 1])

                X1.append(query)
                cX1.append(cquery)

                X2.append(neg_doc)
                cX2.append(cneg_doc)

                y.append([1, 0])

                if i % batch_size == 0 and i != 0:
                    yield ({'question_input': np.array(X1), 'passage_input': np.array(X2),
                            'char_question_input': np.array(cX1), 'char_passage_input': np.array(cX2)}, np.array(y))
                    X1, X2, y = [], [], []
                    cX1, cX2 = [], []


    def _get_pair_list(self, queries, docs, labels):
        """Yields a character and word based indexed pair list of the format
        question, pos_doc, negative_doc

        Parameters
        ----------
        queries : list of list of str
        docs : list of list of list of str
        labels : list of list of list of int

        """
        while True:
            for q, doc, label in zip(queries, docs, labels):
                doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
                for item in zip(doc, label):
                    if item[1] == 1:
                        for new_item in zip(doc, label):
                            if new_item[1] == 0:
                                yield(
                                  self._make_sentence_indexed_padded(q, self.max_question_words),
                                  self._make_sentence_indexed_padded(item[0], self.max_passage_words),
                                  self._make_sentence_indexed_padded(new_item[0], self.max_passage_words),
                                  self._make_sentence_indexed_padded_charred(q, self.max_question_words),
                                  self._make_sentence_indexed_padded_charred(item[0],self.max_passage_words),
                                  self._make_sentence_indexed_padded_charred(new_item[0], self.max_passage_words)
                                )

    def _string2numeric_hash(self, text):
        "Gets a numeric hash for a given string"
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def _seeded_vector(self, seed_string, vector_size):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = np_random.RandomState(self._string2numeric_hash(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size


    def _get_model(self, max_passage_sents, max_passage_words, max_question_words, embedding_matrix, n_highway_layers=2,
            highway_activation='relu', embed_trainable=False, n_encoder_hidden_nodes=200, filters=100, depth=5,
            max_word_charlen=25, char_embedding_dim=8):
        """Returns a keras model as per the BiDAF-T architecture"""

        total_passage_words = max_passage_sents * max_passage_words

        # (batch_size, max_question_words)
        question_input = Input(shape=(max_question_words,), dtype='int32', name="question_input")
        # (batch_size, total_passage_words)
        passage_input = Input(shape=(total_passage_words,), dtype='int32', name="passage_input")

        # (batch_size, max_question_words, max_word_charlen)
        char_question_input = Input(shape=(max_question_words, max_word_charlen), dtype='int32', name="char_question_input")
        # (batch_size, total_passage_words, max_word_charlen)
        char_passage_input = Input(shape=(total_passage_words, max_word_charlen), dtype='int32', name="char_passage_input")

        embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                  weights=[embedding_matrix], trainable=embed_trainable)


        # Get a character embedding for each word
        char_embedding_layer = Embedding(input_dim=len(self.char2index) + 1, output_dim=char_embedding_dim)

        # (batch_size, max_question_words, max_word_charlen, char_embedding_dim)
        char_q = char_embedding_layer(char_question_input)
        # (batch_size, total_passage_words, max_word_charlen, char_embedding_dim)
        char_p = char_embedding_layer(char_passage_input)

        q_conv_layer = TimeDistributed(Conv1D(filters, depth))
        p_conv_layer = TimeDistributed(Conv1D(filters, depth))
        char_q = q_conv_layer(char_q)
        char_p = p_conv_layer(char_p)

        mpool_layer = Lambda(lambda x: tf.reduce_max(x, -2))

        # (batch_size, max_question_words, depth)
        char_q = mpool_layer(char_q)
        # (batch_size, total_passage_words, depth)
        char_p = mpool_layer(char_p)
       

        question_embedding = embedding_layer(question_input)  # (batch_size, max_question_words, embedding_dim)
        passage_embedding = embedding_layer(passage_input)  # (batch_size, total_passage_words, embedding_dim)

        # (batch_size, max_question_words, depth + word_embedding_dim)
        question_embedding = Concatenate()([question_embedding, char_q])
        # (batch_size, total_passage_words, depth + word_embedding_dim)
        passage_embedding = Concatenate()([passage_embedding, char_p])

        # Highway Layer doesn't affect the shape of the tensor
        for i in range(n_highway_layers):
            highway_layer = Highway(activation='relu', name='highway_{}'.format(i))
            
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_embedding)

            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_embedding)

        # To capture contextual information in a passage
        passage_bidir_encoder = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True,
                                                                   name='PassageBidirEncoder'), merge_mode='concat')

        encoded_passage = passage_bidir_encoder(passage_embedding)  # (batch_size, total_passage_words, 2*n_encoder_hidden_nodes)
        encoded_question = passage_bidir_encoder(question_embedding)  # (batch_size, max_question_words, 2*n_encoder_hidden_nodes)

        # Reshape to calculate a dot similarity between query and passage
        
        # (batch_size, total_passage_words, max_question_words, 2*n_encoder_hidden_nodes)
        tiled_passage = Lambda(lambda x: tf.tile(tf.expand_dims(x, 2), [1, 1, max_question_words, 1]))(encoded_passage)

        # (batch_size, total_passage_words, max_question_words, 2*n_encoder_hidden_nodes)
        tiled_question = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, total_passage_words, 1, 1]))(encoded_question)

        # (batch_size, total_passage_words, max_question_words, 2*n_encoder_hidden_nodes)
        a_elmwise_mul_b = Lambda(lambda x:tf.multiply(x[0], x[1]))([tiled_passage, tiled_question])

        # (batch_size, total_passage_words, max_question_words, 6*n_encoder_hidden_nodes)
        concat_data = Concatenate(axis=-1)([tiled_passage, tiled_question, a_elmwise_mul_b])

        S = Dense(1)(concat_data)
        S = Lambda(lambda x: K.squeeze(x, -1))(S)  # (batch_size, total_passage_words, max_question_words)

        # Normalize using softmax for the `max_question_words` dimension
        S = Activation('softmax')(S)

        # Context2Query (Passage2Query)
        # batch_matmul((batch_size, total_passage_words, max_question_words), (batch_size, max_question_words, 2*n_encoder_hidden_nodes) ) =
        # (batch_size, total_passage_words, 2*n_encoder_hidden_nodes)
        c2q = Lambda(lambda x: tf.matmul(x[0], x[1]))([S, encoded_question])

        # Query2Context
        # b: attention weights on the context
        b = Lambda(lambda x: tf.nn.softmax(K.max(x, 2), dim=-1), name='b')(S) # (batch_size, total_passage_words)

        # batch_matmul( (batch_size, 1, total_passage_words), (batch_size, total_passage_words, 2*n_encoder_hidden_nodes) ) = 
        # (batch_size, 1, 2*n_encoder_hidden_nodes)
        q2c = Lambda(lambda x:tf.matmul(tf.expand_dims(x[0], 1), x[1]))([b, encoded_passage]) 
        # (batch_size, total_passage_words, 2*n_encoder_hidden_nodes), tiled `total_passage_words` times
        q2c = Lambda(lambda x: tf.tile(x, [1, total_passage_words, 1]))(q2c)


        # G: query aware representation of each context word
         # (batch_size, total_passage_words, 8*n_encoder_hidden_nodes)
        G = Lambda(lambda x: tf.concat([x[0], x[1], tf.multiply(x[0], x[1]), tf.multiply(x[0], x[2])], axis=-1))([encoded_passage, c2q, q2c])


        # Get some more context based info over the passage words
        modelled_passage = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True))(G)
        modelled_passage = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True))(modelled_passage)

        # Reshape it back to be at the sentence level
        #reshaped_passage = Reshape((max_passage_sents, max_passage_words, n_encoder_hidden_nodes*2))(modelled_passage)

        g2 = Lambda(lambda x: tf.reduce_max(x, 1))(modelled_passage)

        pred = Dense(2, activation='softmax')(g2)

        model = Model(inputs=[question_input, passage_input, char_question_input, char_passage_input], outputs=[pred])
        return model

    def train(self, queries=None, docs=None, labels=None, n_epochs=None, steps_per_epoch=None, batch_size=None):
        """Trains the model on the existing or given queries, docs and labels"""

        # If you're building for the first time
        if self.model is None:
            # Build Model
            self.model = self._get_model(max_passage_sents=self.max_passage_sents, max_passage_words=self.max_passage_words,
                         max_question_words=self.max_question_words, embedding_matrix=self.embedding_matrix, 
                         n_encoder_hidden_nodes=self.n_encoder_hidden_nodes, char_embedding_dim=self.char_embedding_dim, 
                         depth=self.depth, filters=self.filters, max_word_charlen=self.max_word_charlen
                         )
            self.model.summary()
            self.model.compile(loss=self.loss, optimizer=self.optimizer)

        # To allow retraining
        self.queries = queries or self.queries
        self.docs = docs or self.docs
        self.labels = labels or self.labels
        self.n_epochs = n_epochs or self.n_epochs
        self.steps_per_epoch = steps_per_epoch or self.steps_per_epoch
        self.batch_size = batch_size or self.batch_size

        train_generator = self._get_full_batch_iter(self._get_pair_list(self.queries, self.docs, self.labels), self.batch_size)
        self.model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.n_epochs)


    def batch_predict(self, q, doc):
        """Returns predictions on a query and doc batch

        Parameters
        ----------
        q : list of list of str
        doc : lit of list of list of str

        Example
        -------
        q = [['hello', 'there']]
        doc = [['general', 'kenobi'],
             ['i', 'am', 'he']]

        Predictions would be like
        [[0.1, 0.9],  # 90% probablity of this sentence being the correct answer
         [0.7, 0.3]]  # 30% probablity of this sentence being the correct answer

        Effectively, we can rank the answers with the 90% and 30%
        """
        wq, cq = [], []
        test_docs, ctest_docs = [], []
        num_docs = len(doc)

        for d in doc:
            cq.append(self._make_sentence_indexed_padded_charred(q, self.max_question_words))
            wq.append(self._make_sentence_indexed_padded(q, self.max_question_words))
            test_docs.append(self._make_sentence_indexed_padded(d, self.max_passage_words))
            ctest_docs.append(self._make_sentence_indexed_padded_charred(d, self.max_passage_words))

        wq = np.array(wq).reshape((num_docs, self.max_question_words))
        cq = np.array(cq).reshape((num_docs, self.max_question_words, self.max_word_charlen))

        test_docs = np.array(test_docs).reshape((num_docs, self.total_passage_words))
        ctest_docs = np.array(ctest_docs).reshape((num_docs, self.total_passage_words, self.max_word_charlen))

        preds = self.model.predict(x={'question_input':wq,  'passage_input':test_docs,
                                      'char_passage_input': ctest_docs, 'char_question_input': cq})
        return preds

