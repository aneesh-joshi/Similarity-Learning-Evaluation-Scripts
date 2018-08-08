from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed,\
                         LSTM, Bidirectional, Lambda, Reshape, Activation, Masking, Conv1D
from keras.models import Model
from keras.utils.np_utils import to_categorical
import keras.backend as K

from .utils.custom_layers import Highway

import numpy as np
import tensorflow as tf
import os
from collections import Counter
import gensim.downloader as api
import logging
import hashlib
from numpy import random as np_random
import string
from keras import optimizers


logger = logging.getLogger(__name__)

class BiDAF_T:
    def __init__(self, queries, docs, labels, kv_model, max_passage_words=100, max_passage_sents=1, max_question_words=40,
        char_embedding_dim=100, batch_size=50, unk_handle_method='zero', pad_handle_method='zero', optimizer='adam',
        n_epochs=5, n_encoder_hidden_nodes=200, max_word_charlen=25, depth=100, filters=5, word_embedding_dim=100,
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
        self.max_word_charlen = self.max_word_charlen
        self.loss = 'categorical_crossentropy'
        self.n_encoder_hidden_nodes = self.n_encoder_hidden_nodes
        self.char_pad_index = 0
        self.pad_word_index = 0
        self.depth = depth
        self.filters = filters
        #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # ada = optimizers.Adadelta(lr=0.5)
        # optimizer = ada
        self.steps_per_epoch = steps_per_epoch

        self.char2index = {}
        self.word2index = {}



# # Text Preprocessing
# q_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
# d_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
# l_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))

# q_train_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
# d_train_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
# l_train_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))



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

        # import matplotlib.pyplot as plt
        # plt.hist(q_lens)
        # plt.show()
        # plt.hist(doc_lens)
        # plt.show()
        # plt.hist(n_docs_per_query)
        # plt.show()

        self.vocab_size = len(word_counter)
        logger.info('There are %d vocab words in the training data', self.vocab_size)
        logger.info('There are %d queries and %d docs in the training data', n_queries, n_docs)

        self.word2index = {}
        for i, word in enumerate(word_counter.keys(), start=1):
            word2index[word] = i

        self.pad_word_index = 0
        self.unk_word_index = vocab_size + 1

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

        self.embedding_matrix = np.zeros((1 + len(self.kv_model.vocab) + 1 + num_embedding_words , self.kv_model.vector_size))
        # 1 for pad word, 1 for unk_word in non-train data

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
        """Converts a str word into a list of its character indices with pads"""
        list_word = list(word)
        indexed_word = [self.char2index[l] for l in list_word]
        indexed_word = indexed_word + (self.max_word_charlen - len(indexed_word))*[char_pad_index]
        if len(indexed_word) > self.max_word_charlen:
            logger.info('Word: %s is really long of length: %d. Clipping to length %d for now \n\n\n',
                         word, len(indexed_word), self.max_word_charlen)
            indexed_word = indexed_word[:self.max_word_charlen]
        return indexed_word

    def _make_sentence_indexed_padded_charred(self, sentence, max_len):
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
        sentence = [self._char_word(word) for word in sentence]
        while len(sentence) < max_len:
            sentence += [[self.char_pad_index] * self.max_word_charlen]
        if len(sentence) > max_len:
            logger.info("Max length of %d wasn't long enough for sentence:%s of length %d. Truncating sentence for now",
                max_len, str(str_sent), len(sentence))
            sentence = sentence[:max_len]
        return sentence


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
            logger.info("Max length of %d wasn't long enough for sentence:%s of length %d. Truncating sentence for now",
                max_len, str(str_sent), len(sentence))
            sentence = sentence[:max_len]
        return sentence

    def train_batch_generator(self, queries, docs, labels, batch_size):
        '''Yields a batch for training
        query: (batch_size, max_question_words)
        docs: (batch_size, max_passage_sents*max_passage_words)
        label: (batch_size, max_passage_sents, 1, 2)
        '''
        while True:
            cbatch_q, cbatch_d, cbatch_l = [], [], []
            ctrain_qs, ctrain_ds, ctrain_ls = [], [], []
            batch_q, batch_d, batch_l = [], [], []
            train_qs, train_ds, train_ls = [], [], []
            
            for i, (q, docs, labels) in enumerate(zip(queries, docs, labels)):

                if i % self.batch_size == 0 and i != 0:
                    bq, bd, bl = np.array(batch_q), np.array(batch_d), np.array(batch_l)
                    cbq, cbd, cbl = np.array(cbatch_q), np.array(cbatch_d), np.array(cbatch_l)

                    bd = bd.reshape((-1, total_passage_words))
                    bl = np.squeeze(bl)
                    bq = np.squeeze(bq)

                    cbd = cbd.reshape((-1, total_passage_words, max_word_charlen))
                    cbl = np.squeeze(cbl)
                    cbq = np.squeeze(cbq)

                    yield ({'question_input': bq, 'passage_input': bd, 'char_question_input': cbq, 'char_passage_input': cbd}, bl)
                    batch_q, batch_d, batch_l = [], [], []
                    cbatch_q, cbatch_d, cbatch_l = [], [], []

                for d, l in zip(docs, labels):
                    train_qs.append(_make_sentence_indexed_padded(q, max_question_words))
                    ctrain_qs.append(_make_sentence_indexed_padded_charred(q, max_question_words))
                    train_ds.append(_make_sentence_indexed_padded(d, self.max_passage_words))
                    ctrain_ds.append(_make_sentence_indexed_padded_charred(d, self.max_passage_words))
                    train_ls.append(to_categorical(l, 2))

                # Add extra sentences in the passage to make it of the same length and set them to false
                while(len(train_ds) < self.max_passage_sents):
                    train_ds.append([self.pad_word_index] * self.max_passage_words)
                    ctrain_ds.append([[char_pad_index]*max_word_charlen] * self.max_passage_words)
                    train_ls.append(to_categorical(0, 2))

                if len(train_ds) > self.max_passage_sents:
                    raise ValueError("%d max_passage_sents isn't long enough for num docs %d" % (self.max_passage_sents, len(train_ds)))

                batch_q.append(train_qs)
                batch_d.append(train_ds)
                batch_l.append(train_ls)

                cbatch_q.append(ctrain_qs)
                cbatch_d.append(ctrain_ds)
                cbatch_l.append(ctrain_ls)

                train_qs, train_ds, train_ls = [], [], []
                ctrain_qs, ctrain_ds, ctrain_ls = [], [], []
            logger.info('One epoch worth of samples are exhausted')

    def _get_full_batch_iter(self, pair_list, batch_size):
        """Returns batches with alternate positive and negative docs taken from
        the `pair_list`
        """
        X1, X2, y = [], [], []
        cX1, cX2 = [], []
        batch_size = batch_size/2
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
        while True:
            for q, doc, label in zip(queries, docs, labels):
                doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
                for item in zip(doc, label):
                    if item[1] == 1:
                        for new_item in zip(doc, label):
                            if new_item[1] == 0:
                                yield(
                                  _make_sentence_indexed_padded(q, max_question_words),
                                  _make_sentence_indexed_padded(item[0], self.max_passage_words),
                                  _make_sentence_indexed_padded(new_item[0], self.max_passage_words),
                                  _make_sentence_indexed_padded_charred(q, max_question_words),
                                  _make_sentence_indexed_padded_charred(item[0],self.max_passage_words),
                                  _make_sentence_indexed_padded_charred(new_item[0], self.max_passage_words)
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
            highway_activation='relu', embed_trainable=False, n_encoder_hidden_nodes=200, filters=100, depth=5):
        
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
            highway_layer = Highway(activation=highway_activation, name='highway_{}'.format(i))
            
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

    def train(self):
        # Build Model
        self.model = self._get_model(self.max_passage_sents, self.max_passage_words, self.max_question_words,
                        self.embedding_matrix, self.n_encoder_hidden_nodes, self.char_embedding_dim, self.depth,
                        self.filters)
        self.model.summary()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch, epochs=self.n_epochs)

        # train_generator = self._get_full_batch_iter(self._get_pair_list(self.queries, self.docs, self.labels), self.batch_size)

        # print("Training on WikiQA now")
        # #train_generator = train_batch_generator(q_train_iterable, d_train_iterable, l_train_iterable, batch_size)
        # train_generator = _get_full_batch_iter(_get_pair_list(q_train_iterable, d_train_iterable, l_train_iterable), batch_size)
        # model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)

        # Evaluation

        # queries, doc_group, label_group, query_ids, doc_id_group = MyOtherWikiIterable(
        #                                                                 os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv')
        #                                                             ).get_stuff()

    def batch_predict(self, model, q, doc):

        wq, cq = [], []
        test_docs, ctest_docs = [], []
        num_docs = len(doc)

        for d in doc:
            cq.append(self._make_sentence_indexed_padded_charred(q, self.max_question_words))
            wq.append(self._make_sentence_indexed_padded(q, self.max_question_words))
            test_docs.append(self._make_sentence_indexed_padded(d, self.max_passage_words))
            ctest_docs.append(self._make_sentence_indexed_padded_charred(d, self.max_passage_words))

        wq = np.array(wq).reshape((num_docs, self.max_question_words))
        cq = np.array(cq).reshape((num_docs, self.max_question_words, max_word_charlen))

        test_docs = np.array(test_docs).reshape((num_docs, total_passage_words))
        ctest_docs = np.array(ctest_docs).reshape((num_docs, total_passage_words, max_word_charlen))

        preds = model.predict(x={'question_input':wq,  'passage_input':test_docs,
                                 'char_passage_input': ctrain_docs, 'char_question_input': cq})

        print(preds)
        return preds

# def test_generator():
#     batch_q, batch_d, batch_l = [], [], []
#     train_qs, train_ds, train_ls = [], [], []
#     final_dlens, final_q, final_d = [], [], []
#     doc_lens = []
#     for i, (q, docs, labels) in enumerate(zip(queries, doc_group, label_group)):
        
#         if i % batch_size == 0 and i != 0:
#             bq, bd, bl = np.array(batch_q), np.array(batch_d), np.array(batch_l)
#             bd = bd.reshape((-1, total_passage_words))
#             bl = np.squeeze(bl)
#             bq = np.squeeze(bq)
#             final_q.append(bq)
#             final_d.append(bd)
#             final_dlens.append(doc_lens)
#             doc_lens = []
#             batch_q, batch_d, batch_l = [], [], []
#         train_qs.append(_make_sentence_indexed_padded(q, max_question_words))

#         for d, l in zip(docs, labels):
#             train_ds.append(_make_sentence_indexed_padded(d, max_passage_words))
#             train_ls.append(to_categorical(l, 2))
        
#         doc_lens.append(len(train_ds))
#         # Add extra sentences in the passage to make it of the same length and set them to false
#         while(len(train_ds) < max_passage_sents):
#             train_ds.append([self.pad_word_index] * max_passage_words)
#             train_ls.append(to_categorical(0, 2))

#         if len(train_ds) > max_passage_sents:
#             raise ValueError("%d max_passage_sents isn't long enough for num docs %d" % (max_passage_sents, len(train_ds)))

#         batch_q.append(train_qs)
#         batch_d.append(train_ds)
#         batch_l.append(train_ls)

#         train_qs, train_ds, train_ls = [], [], []
    
#     return np.array(final_q), np.array(final_d), np.array(final_dlens)


