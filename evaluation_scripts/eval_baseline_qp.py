import sys
import os
sys.path.append('..')
from sl_eval.models import BaselineModel
from gensim import downloader as api
import numpy as np
from keras.utils import to_categorical
import random
import re
from sklearn.utils import shuffle

random.seed(42)  # seed the shuffle
    
def sent2vec(sent):
    if len(sent)==0:
        print('length is 0, Returning random')
        return np.random.random((kv_model.vector_size,))

    vec = []
    for word in sent:
        if word in kv_model:
            vec.append(kv_model[word])

    if len(vec) == 0:
        print('No words in vocab, Returning random')
        return np.random.random((kv_model.vector_size,))

    vec = np.array(vec)

    return np.mean(vec, axis=0)


def preprocess(sent):
    return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

if __name__ == '__main__':

    print('Evaluating Quora Duplicate Questions Baseline')
    num_predictions = 2
    num_embedding_dims = 300
    kv_model = api.load('glove-wiki-gigaword-' + str(num_embedding_dims))

    train_split = 0.8
    num_samples = 323432

    qqp = api.load('quora-duplicate-questions')

    sent_len = []

    q1, q2, duplicate = [], [], []
    for row in qqp:
        sent_len.append(len(row['question1']))
        sent_len.append(len(row['question2']))
        q1.append(preprocess(row['question1']))
        q2.append(preprocess(row['question2']))
        duplicate.append(int(row['is_duplicate']))

    print('Average sentence length is ' + str(sum(sent_len)/len(sent_len)))

    print('Number of question pairs', len(q1))
    print('Number of duplicates', sum(duplicate))
    print('%% duplicates', 100.*sum(duplicate)/len(q1))
    print('-----------------------------------------')

    q1, q2, duplicate = shuffle(q1, q2, duplicate)

    train_q1, test_q1 = q1[:int(len(q1) * train_split)], q1[int(len(q1) * train_split):]
    train_q2, test_q2 = q2[:int(len(q2) * train_split)], q2[int(len(q2) * train_split):]
    train_duplicate, test_duplicate = duplicate[:int(len(duplicate) * train_split)], duplicate[int(len(duplicate) * train_split):]

    assert len(train_q1) == len(train_duplicate)
    assert len(test_q2) == len(test_duplicate)


    print('Number of question pairs in train', len(train_q1))
    print('Number of duplicates in train', sum(train_duplicate))
    print('%% duplicates', 100.*sum(train_duplicate)/len(train_q1))
    print('-----------------------------------------')

    print('Number of question pairs in test', len(test_q1))
    print('Number of duplicates in test', sum(test_duplicate))
    print('%% duplicates', 100.*sum(test_duplicate)/len(test_q1))
    print('-----------------------------------------')



    vectored_train_q1 = [sent2vec(qi) for qi in train_q1]
    vectored_train_q1 = np.array(vectored_train_q1)

    vectored_train_q2 = [sent2vec(qi) for qi in train_q2]
    vectored_train_q2 = np.array(vectored_train_q2)

    one_hot_train_labels = [to_categorical(li, num_predictions) for li in train_duplicate]
    one_hot_train_labels = np.squeeze(np.array(one_hot_train_labels))


    vectored_test_q1 = [sent2vec(qi) for qi in test_q1]
    vectored_test_q1 = np.array(vectored_test_q1)

    vectored_test_q2 = [sent2vec(qi) for qi in test_q2]
    vectored_test_q2 = np.array(vectored_test_q2)

    one_hot_test_labels = [to_categorical(li, num_predictions) for li in test_duplicate]
    one_hot_test_labels = np.squeeze(np.array(one_hot_test_labels))

    regression_model = BaselineModel(
                        vector_size=num_embedding_dims, num_predictions=num_predictions, model_type='regression'
                        )
    regression_model.train({'x1': vectored_train_q1, 'x2': vectored_train_q2}, one_hot_train_labels, n_epochs=10)

    print('\nAccuracy of the Regression Model is:' +
            str(regression_model.model.evaluate({'x1': vectored_test_q1, 'x2': vectored_test_q2},
                                  one_hot_test_labels)[1]
            )
        )

    del regression_model

    multilayer_model = BaselineModel(
                        vector_size=num_embedding_dims, num_predictions=num_predictions, model_type='multilayer'
                        )
    multilayer_model.train({'x1': vectored_train_q1, 'x2': vectored_train_q2}, one_hot_train_labels, n_epochs=10)

    print('\nAccuracy of the Multilayer Model is:' +
            str(multilayer_model.model.evaluate({'x1': vectored_test_q1, 'x2': vectored_test_q2},
                                  one_hot_test_labels)[1]
            )
        )
