import sys
import os
sys.path.append('..')
from sl_eval.models import BaselineModel
from data_readers import SickReader
from gensim import downloader as api
import numpy as np
    
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


if __name__ == '__main__':
    sick_folder_path = os.path.join('..', 'data', 'SICK')
    sick_reader = SickReader(sick_folder_path)

    kv_model = api.load('glove-wiki-gigaword-100')


    x1, x2, label = sick_reader.get_entailment_data()
    train_x1, train_x2, train_labels = x1['TRAIN'], x2['TRAIN'], label['TRAIN']
    test_x1, test_x2, test_labels = x1['TEST'], x2['TEST'], label['TEST']

    print(train_x1[9], sent2vec(train_x1[9]))


    vectored_x1 = [sent2vec(xi) for xi in train_x1]
    vectored_x1 = np.array(vectored_x1)
    print(vectored_x1.shape)

