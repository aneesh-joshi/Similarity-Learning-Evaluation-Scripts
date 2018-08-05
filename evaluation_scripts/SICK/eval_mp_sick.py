import sys
import os
sys.path.append('../..')

from data_readers import SickReader
import gensim.downloader as api
from sl_eval.models.matchpyramid import MatchPyramid

sick_folder_path = os.path.join('..', '..', 'data', 'SICK')
sick_reader = SickReader(sick_folder_path)


x1, x2, label = sick_reader.get_entailment_data()
train_x1, train_x2, train_labels = x1['TRAIN'], x2['TRAIN'], label['TRAIN']
test_x1, test_x2, test_labels = x1['TEST'], x2['TEST'], label['TEST']


word_embedding_len = 50
kv_model = api.load('glove-wiki-gigaword-' + str(word_embedding_len))


batch_size = 50
text_maxlen = 200
n_epochs = 5 
steps_per_epoch = len(train_x1)//batch_size

dtks_model = MatchPyramid(queries=train_x1, docs=train_x2, labels=train_labels, target_mode='inference',
                     word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
dtks_model.evaluate_inference(test_x1, test_x2, test_labels)

