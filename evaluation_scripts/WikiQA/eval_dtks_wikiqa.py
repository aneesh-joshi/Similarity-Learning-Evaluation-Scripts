import sys
sys.path.append('../..')
import os
from sl_eval.models import DRMM_TKS
import gensim.downloader as api
from utils import MyWikiIterable


model_save_path = 'saved_models'
model_name = 'dtks_wikiqa_model'

wikiqa_folder = '..', '..', 'data', 'WikiQACorpus'

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

# Train the model
drmm_tks_model = DRMM_TKS(
                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=3,
                    validation_data=[q_val_iterable, d_val_iterable, l_val_iterable], topk=20
                )

print('Test set results')
drmm_tks_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)

drmm_tks_model.save(os.path.join(model_save_path, model_name))
