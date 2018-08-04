import sys
import os
sys.path.append('../..')

from data_readers import SnliReader	
import gensim.downloader as api
from sl_eval.models.drmm_tks import DRMM_TKS

snli_folder_path = os.path.join('..', '..', 'data', 'snli_1.0', 'snli_1.0')
snli_reader = SnliReader(snli_folder_path)

train_x1, train_x2, train_labels, train_annotator_labels = snli_reader.get_data('train')

word_embedding_len = 300
kv_model = api.load('glove-wiki-gigaword-' + str(word_embedding_len))

batch_size = 10
text_maxlen = 200
n_epochs = 10 
steps_per_epoch = len(train_x1)//batch_size

dtks_model = DRMM_TKS(queries=train_x1, docs=train_x2, labels=train_labels, target_mode='inference',
                     word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
dtks_model.evaluate_inference(test_q1, test_q2, test_duplicate)

