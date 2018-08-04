import sys
sys.path.append('../..')
import sys
import os

from sl_eval.models import MatchPyramid
import gensim.downloader as api
from utils import MyWikiIterable



wikiqa_folder = os.path.join('..', '..', 'data', 'WikiQACorpus')

q_iterable = MyWikiIterable('query', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))
d_iterable = MyWikiIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))
l_iterable = MyWikiIterable('label', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))

q_val_iterable = MyWikiIterable('query', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))
d_val_iterable = MyWikiIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))
l_val_iterable = MyWikiIterable('label', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))

q_test_iterable = MyWikiIterable('query', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))
d_test_iterable = MyWikiIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))
l_test_iterable = MyWikiIterable('label', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))



kv_model = api.load('glove-wiki-gigaword-300')

n_epochs = 2
batch_size = 10
steps_per_epoch = 9000 // batch_size

# Train the model
mp_model = MatchPyramid(
                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
batch_size=batch_size, text_maxlen=200 #validation_data=[q_val_iterable, d_val_iterable, l_val_iterable],
                )

print('Test set results')
mp_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)
model_save_path = 'saved_models'
mp_model.save(os.path.join(model_save_path, 'my_mp_mpdel'))
