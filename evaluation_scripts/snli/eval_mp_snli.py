import sys
import os
sys.path.append('../..')

from data_readers import SnliReader	
import gensim.downloader as api
from sl_eval.models.matchpyramid import MatchPyramid

snli_folder_path = os.path.join('..', '..', 'data', 'snli_1.0')
snli_reader = SnliReader(snli_folder_path)

train_x1, train_x2, train_labels, train_annotator_labels = snli_reader.get_data('train')
test_x1, test_x2, test_labels, test_annotator_labels = snli_reader.get_data('test')

word_embedding_len = 300
kv_model = api.load('glove-wiki-gigaword-' + str(word_embedding_len))

print('There are %d training samples' % len(train_x1))

batch_size = 50
text_maxlen = 200
n_epochs = 5 
steps_per_epoch = len(train_x1)//batch_size

dtks_model = MatchPyramid(queries=train_x1, docs=train_x2, labels=train_labels, target_mode='inference',
                     word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
dtks_model.evaluate_inference(test_x1, test_x2, test_labels)

