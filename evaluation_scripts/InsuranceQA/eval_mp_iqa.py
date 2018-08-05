import sys
import os
sys.path.append('../..')

from data_readers import IQAReader
import gensim.downloader as api
from sl_eval.models.matchpyramid import MatchPyramid

iqa_folder_path = os.path.join('..', '..', 'data', 'insurance_qa_python')
iqa_reader = IQAReader(iqa_folder_path)


# PARAMETERS ---------------------------------------------------------
train_batch_size = 10
test_batch_size = 10
word_embedding_len = 300
batch_size = 50
text_maxlen = 200
n_epochs = 5 
# --------------------------------------------------------------------




train_q, train_d, train_l = iqa_reader.get_train_data(batch_size=4)
test1_q, test1_d, test1_l = iqa_reader.get_test_data('test1', batch_size=4)

kv_model = api.load('glove-wiki-gigaword-' + str(word_embedding_len))
steps_per_epoch = len(train_x1)//batch_size

dtks_model = MatchPyramid(queries=train_q, docs=train_d, labels=train_l, target_mode='ranking',
                     word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size,
                     steps_per_epoch=steps_per_epoch)

dtks_model.evaluate_inference(test_x1, test_x2, test_labels)

