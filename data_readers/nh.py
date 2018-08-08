from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, LSTM, Bidirectional, Lambda, Reshape, Activation, Masking, Conv1D
from keras.models import Model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from custom_layers import Highway

from my_try import MyWikiIterable, MyOtherWikiIterable

import numpy as np
import tensorflow as tf
import os
from collections import Counter
import gensim.downloader as api
import logging
import hashlib
from numpy import random as np_random


logger = logging.getLogger(__name__)

# PARMATERS -------------------------------------------------------
max_passage_words = 100
max_passage_sents = 1
total_passage_words = max_passage_sents * max_passage_words
max_question_words = 40
num_word_embedding_dims = 100

batch_size = 50
unk_handle_method = 'zero'
pad_handle_method = 'zero'

char_embedding_dim = 100

from keras import optimizers
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
ada = optimizers.Adadelta(lr=0.5)
loss = 'categorical_crossentropy'
optimizer = ada
steps_per_epoch = 1087
n_epochs = 5
n_encoder_hidden_nodes = 200

import string
char2index = {}
for i, char in enumerate(string.printable, start=1):
    char2index[char] = i

max_word_charlen = 25
char_pad_index = 0

pretrain = 'SQUAD'
test = 'WikiQA'
# -----------------------------------------------------------------


# Text Preprocessing
q_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
d_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))
l_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'SQUAD-T-QA.tsv'))

q_train_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
d_train_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
l_train_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))

n_queries, n_docs = 0, 0
q_lens, doc_lens, n_docs_per_query = [], [], []

# Create word indexes from training data
word_counter = Counter()
for q, docs, labels in zip(q_iterable, d_iterable, l_iterable):
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

vocab_size = len(word_counter)
logger.info('There are %d vocab words in the training data', vocab_size)
logger.info('There are %d queries and %d docs in the training data', n_queries, n_docs)

word2index = {}
for i, word in enumerate(word_counter.keys(), start=1):
    word2index[word] = i

pad_word_index = 0
unk_word_index = vocab_size + 1

logger.info('The pad_word is set to the index: %d and the unkwnown words will be set to the index %d', pad_word_index, unk_word_index)


def _char_word(word):
    l_word = list(word)
    l_word = [char2index[l] for l in l_word]
    l_word = l_word + (max_word_charlen - len(l_word))*[char_pad_index]
    if len(l_word) > max_word_charlen:
        logger.info('\n\n\n word is really long %d \n\n\n', len(l_word))
        l_word = l_word[:max_word_charlen]
    return l_word

def _make_sentence_indexed_padded_charred(sentence, max_len):
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
    sentence = [_char_word(word) for word in sentence]
    while len(sentence) < max_len:
        sentence += [[char_pad_index]*max_word_charlen]
    if len(sentence) > max_len:
        logger.info("Max length of %d wasn't long enough for sentence:%s of length %d. Truncating sentence for now",
            max_len, str(str_sent), len(sentence))
        sentence = sentence[:max_len]
    return sentence




def _make_sentence_indexed_padded(sentence, max_len):
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
    sentence = [word2index[word] if word in word2index else unk_word_index for word in sentence]
    while len(sentence) < max_len:
        sentence.append(pad_word_index)
    if len(sentence) > max_len:
        logger.info("Max length of %d wasn't long enough for sentence:%s of length %d. Truncating sentence for now",
            max_len, str(str_sent), len(sentence))
        sentence = sentence[:max_len]
    return sentence

def train_batch_generator(q_iterable, d_iterable, l_iterable, batch_size):
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
        
        for i, (q, docs, labels) in enumerate(zip(q_iterable, d_iterable, l_iterable)):

            if i % batch_size == 0 and i != 0:
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
                train_ds.append(_make_sentence_indexed_padded(d, max_passage_words))
                ctrain_ds.append(_make_sentence_indexed_padded_charred(d, max_passage_words))
                train_ls.append(to_categorical(l, 2))

            # Add extra sentences in the passage to make it of the same length and set them to false
            while(len(train_ds) < max_passage_sents):
                train_ds.append([pad_word_index] * max_passage_words)
                ctrain_ds.append([[char_pad_index]*max_word_charlen] * max_passage_words)
                train_ls.append(to_categorical(0, 2))

            if len(train_ds) > max_passage_sents:
                raise ValueError("%d max_passage_sents isn't long enough for num docs %d" % (max_passage_sents, len(train_ds)))

            batch_q.append(train_qs)
            batch_d.append(train_ds)
            batch_l.append(train_ls)

            cbatch_q.append(ctrain_qs)
            cbatch_d.append(ctrain_ds)
            cbatch_l.append(ctrain_ls)

            train_qs, train_ds, train_ls = [], [], []
            ctrain_qs, ctrain_ds, ctrain_ls = [], [], []
        logger.info('One epoch worth of samples are exhausted')

def _get_full_batch_iter(pair_list, batch_size):
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


def _get_pair_list(queries, docs, labels):
    while True:
        for q, doc, label in zip(queries, docs, labels):
            doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
            for item in zip(doc, label):
                if item[1] == 1:
                    for new_item in zip(doc, label):
                        if new_item[1] == 0:
                            yield(_make_sentence_indexed_padded(q, max_question_words), _make_sentence_indexed_padded(item[0], max_passage_words), _make_sentence_indexed_padded(new_item[0], max_passage_words),
                                  _make_sentence_indexed_padded_charred(q, max_question_words), _make_sentence_indexed_padded_charred(item[0], max_passage_words), _make_sentence_indexed_padded_charred(new_item[0], max_passage_words),)




def _string2numeric_hash(text):
    "Gets a numeric hash for a given string"
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

def _seeded_vector( seed_string, vector_size):
    """Create one 'random' vector (but deterministic by seed_string)"""
    # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
    once = np_random.RandomState(_string2numeric_hash(seed_string) & 0xffffffff)
    return (once.rand(vector_size) - 0.5) / vector_size

# Make embedding matrix
kv_model = api.load('glove-wiki-gigaword-' + str(num_word_embedding_dims))


num_embedding_words = 0
num_non_embedding_words = 0
for word in word2index:
    if word in kv_model:
        num_embedding_words += 1
    else:
        num_non_embedding_words += 1

logger.info("There are %d words in the embdding matrix and %d which aren't there", num_embedding_words, num_non_embedding_words)

embedding_matrix = np.zeros((1 + len(kv_model.vocab) + 1 + num_embedding_words , kv_model.vector_size))
# 1 for pad word, 1 for unk_word in non-train data

if pad_handle_method == 'zero':
    embedding_matrix[pad_word_index] = np.zeros((kv_model.vector_size))
elif pad_handle_method == 'random':
    embedding_matrix[pad_word_index] = np.random.random((kv_model.vector_size))
else:
    raise ValueError()

if unk_handle_method == 'zero':
    embedding_matrix[unk_word_index] = np.zeros((kv_model.vector_size))
elif unk_handle_method == 'random':
    embedding_matrix[unk_word_index] = np.random.random((kv_model.vector_size))
else:
    raise ValueError()

for word in word2index:
    if word in kv_model:
        embedding_matrix[word2index[word]] = kv_model[word]
    else:
        embedding_matrix[word2index[word]] = _seeded_vector(word, kv_model.vector_size)

additional_embedding_index = pad_word_index + 1
for word in kv_model.vocab:
    if word not in word2index:
        embedding_matrix[additional_embedding_index] = kv_model[word]
        word2index[word] = additional_embedding_index
        additional_embedding_index += 1


def get_model(max_passage_sents, max_passage_words, max_question_words, embedding_matrix, n_highway_layers = 2,
        highway_activation = 'relu', embed_trainable=False, n_encoder_hidden_nodes = 200, ):
    
    total_passage_words = max_passage_sents * max_passage_words

    question_input = Input(shape=(max_question_words,), dtype='int32', name="question_input")
    passage_input = Input(shape=(total_passage_words,), dtype='int32', name="passage_input")

    char_question_input = Input(shape=(max_question_words, max_word_charlen), dtype='int32', name="char_question_input")
    char_passage_input = Input(shape=(total_passage_words, max_word_charlen), dtype='int32', name="char_passage_input")

    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                              weights=[embedding_matrix], trainable=embed_trainable)

    char_embedding_layer = Embedding(input_dim=len(char2index) + 1, output_dim=char_embedding_dim)
    char_q = char_embedding_layer(char_question_input)
    char_p = char_embedding_layer(char_passage_input)
    filters, depth = 100, 5
    q_conv_layer = TimeDistributed(Conv1D(filters, depth))
    p_conv_layer = TimeDistributed(Conv1D(filters, depth))
    char_q = q_conv_layer(char_q)
    char_p = p_conv_layer(char_p)

    mpool_layer = Lambda(lambda x: tf.reduce_max(x, -2))
    char_q = mpool_layer(char_q)
    char_p = mpool_layer(char_p)
   

    question_embedding = embedding_layer(question_input)  # (batch_size, max_question_words, embedding_dim)
    passage_embedding = embedding_layer(passage_input)  # (batch_size, total_passage_words, embedding_dim)

    question_embedding = Concatenate()([question_embedding, char_q])
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

# Build Model
model = get_model(max_passage_sents, max_passage_words, max_question_words, embedding_matrix, n_encoder_hidden_nodes=n_encoder_hidden_nodes)
model.summary()
model.compile(loss=loss, optimizer=optimizer)

#train_generator = train_batch_generator(q_iterable, d_iterable, l_iterable, batch_size)
train_generator = _get_full_batch_iter(_get_pair_list(q_iterable, d_iterable, l_iterable), batch_size)
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)

print("Training on WikiQA now")
#train_generator = train_batch_generator(q_train_iterable, d_train_iterable, l_train_iterable, batch_size)
train_generator = _get_full_batch_iter(_get_pair_list(q_train_iterable, d_train_iterable, l_train_iterable), batch_size)
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)

# Evaluation

queries, doc_group, label_group, query_ids, doc_id_group = MyOtherWikiIterable(
                                                                os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv')
                                                            ).get_stuff()

'''
batch_q, batch_d, batch_l = [], [], []
train_qs, train_ds, train_ls = [], [], []
for i, (q, docs, labels) in enumerate(zip(queries, doc_group, label_group)):
    train_qs.append(_make_sentence_indexed_padded(q, max_question_words))
    for d, l in zip(docs, labels):
        train_ds.append(_make_sentence_indexed_padded(d, max_passage_words))
        train_ls.append(to_categorical(l, 2))

    # Add extra sentences in the passage to make it of the same length and set them to false
    while(len(train_ds) < max_passage_sents):
        train_ds.append([pad_word_index] * max_passage_words)
        train_ls.append(to_categorical(0, 2))

    
    if len(train_ds) > max_passage_sents:
        raise ValueError("%d max_passage_sents isn't long enough for num docs %d" % (max_passage_sents, len(train_ds)))

    batch_q = train_qs
    batch_d.append(train_ds)
    batch_l.append(train_ls)

    train_ds, train_ls = [], []


bq, bd, bl = np.array(batch_q), np.array(batch_d), np.array(batch_l)
bd = bd.reshape((-1, total_passage_words))
bl = np.squeeze(bl)
test_data = ({'question_input': bq, 'passage_input': bd}, bl)

# print(test_data[0]['passage_input'][0])
# print(test_data[0]['question_input'][0])
# print(test_data[1].shape)
'''

def test_generator():
    batch_q, batch_d, batch_l = [], [], []
    train_qs, train_ds, train_ls = [], [], []
    final_dlens, final_q, final_d = [], [], []
    doc_lens = []
    for i, (q, docs, labels) in enumerate(zip(queries, doc_group, label_group)):
        
        if i % batch_size == 0 and i != 0:
            bq, bd, bl = np.array(batch_q), np.array(batch_d), np.array(batch_l)
            bd = bd.reshape((-1, total_passage_words))
            bl = np.squeeze(bl)
            bq = np.squeeze(bq)
            final_q.append(bq)
            final_d.append(bd)
            final_dlens.append(doc_lens)
            doc_lens = []
            batch_q, batch_d, batch_l = [], [], []
        train_qs.append(_make_sentence_indexed_padded(q, max_question_words))

        for d, l in zip(docs, labels):
            train_ds.append(_make_sentence_indexed_padded(d, max_passage_words))
            train_ls.append(to_categorical(l, 2))
        
        doc_lens.append(len(train_ds))
        # Add extra sentences in the passage to make it of the same length and set them to false
        while(len(train_ds) < max_passage_sents):
            train_ds.append([pad_word_index] * max_passage_words)
            train_ls.append(to_categorical(0, 2))

        if len(train_ds) > max_passage_sents:
            raise ValueError("%d max_passage_sents isn't long enough for num docs %d" % (max_passage_sents, len(train_ds)))

        batch_q.append(train_qs)
        batch_d.append(train_ds)
        batch_l.append(train_ls)

        train_qs, train_ds, train_ls = [], [], []
    
    return np.array(final_q), np.array(final_d), np.array(final_dlens)

def batch_tiny_predict(model, q, doc):
    """To make a prediction on on query and a batch of docs
    Typically speeds up prediction

    Parameters
    ----------
    q : str
    d : list of str
    """
    
    cq = [_make_sentence_indexed_padded_charred(q, max_question_words)]
    q = [_make_sentence_indexed_padded(q, max_question_words)]
    train_docs, ctrain_docs = [], []
    for d in doc:
        train_docs.append(_make_sentence_indexed_padded(d, max_passage_words))
        ctrain_docs.append(_make_sentence_indexed_padded_charred(d, max_passage_words))

    d_len = len(train_docs)

    while(len(train_docs) < max_passage_sents):
        train_docs.append([pad_word_index] * max_passage_words)
        ctrain_docs.append([[char_pad_index]*max_word_charlen] * max_passage_words)

    q = np.array(q).reshape((1, max_question_words))
    cq = np.array(cq).reshape((1, max_question_words, max_word_charlen))
    train_docs = np.array(train_docs).reshape((1, total_passage_words))
    try:
        ctrain_docs = np.array(ctrain_docs).reshape((1, total_passage_words, max_word_charlen))
    except:
        print(ctrain_docs)
        exit()
    preds = model.predict(x={'question_input':q,  'passage_input':train_docs, 'char_passage_input': ctrain_docs, 'char_question_input': cq})

    return preds[0][:d_len]

def new_batch_tiny_predict(model, q, doc):

    wq, cq = [], []
    train_docs, ctrain_docs = [], []

    num_docs = len(doc)

    for d in doc:
        cq.append(_make_sentence_indexed_padded_charred(q, max_question_words))
        wq.append(_make_sentence_indexed_padded(q, max_question_words))
        train_docs.append(_make_sentence_indexed_padded(d, max_passage_words))
        ctrain_docs.append(_make_sentence_indexed_padded_charred(d, max_passage_words))

    wq = np.array(wq).reshape((num_docs, max_question_words))
    cq = np.array(cq).reshape((num_docs, max_question_words, max_word_charlen))

    train_docs = np.array(train_docs).reshape((num_docs, total_passage_words))
    ctrain_docs = np.array(ctrain_docs).reshape((num_docs, total_passage_words, max_word_charlen))

    preds = model.predict(x={'question_input':wq,  'passage_input':train_docs, 'char_passage_input': ctrain_docs, 'char_question_input': cq})

    print(preds)

    return preds

i=0
with open('jpred', 'w') as f:
    for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
        batch_score = new_batch_tiny_predict(model, q, doc)
        for d, l, d_id, bscore in zip(doc, labels, d_ids, batch_score):
            #print(bscore)
            my_score = bscore[1]
            print(i, my_score)
            i += 1
            f.write(q_id + '\t' + 'Q0' + '\t' + str(d_id) + '\t' + '99' + '\t' + str(my_score) + '\t' + 'STANDARD' + '\n')
print("Prediction done. Saved as %s" % 'jpred')

with open('qrels', 'w') as f:
    for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
        for d, l, d_id in zip(doc, labels, d_ids):
            f.write(q_id + '\t' +  '0' + '\t' +  str(d_id) + '\t' + str(l) + '\n')
print("qrels done. Saved as %s" % 'qrels')


'''
q, d, d_lens = test_generator()
print(q.shape, d.shape, d_lens.shape)
print(d_lens)
i=0
for t1, t2, dl in zip(q, d, d_lens):
    test_preds = model.predict(x={'question_input': t1, 'passage_input': t2}, batch_size=batch_size)
    for t, _dl in zip(test_preds, dl):
        print(t[:_dl])
    print('===================')'''
