import sys
sys.path.append('../..')
import sys
import os
import csv
import re
from sl_eval.models import MatchPyramid
import gensim.downloader as api

class MyWikiIterable:
    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def __iter__(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        LABEL_INDEX = 6

        document_group = []
        label_group = []

        n_relevant_docs = 0
        n_filtered_docs = 0

        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                    else:
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []

            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0

q_iterable = MyWikiIterable('query', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))
d_iterable = MyWikiIterable('doc', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))
l_iterable = MyWikiIterable('label', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))

q_val_iterable = MyWikiIterable('query', os.path.join('..', '..',  'data', 'WikiQACorpus', 'WikiQA-dev.tsv'))
d_val_iterable = MyWikiIterable('doc', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-dev.tsv'))
l_val_iterable = MyWikiIterable('label', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-dev.tsv'))

q_test_iterable = MyWikiIterable('query', os.path.join('..', '..',  'data', 'WikiQACorpus', 'WikiQA-test.tsv'))
d_test_iterable = MyWikiIterable('doc', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-test.tsv'))
l_test_iterable = MyWikiIterable('label', os.path.join('..', '..', 'data', 'WikiQACorpus', 'WikiQA-test.tsv'))


kv_model = api.load('glove-wiki-gigaword-300')

# Train the model
mp_model = MatchPyramid(
                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=6, text_maxlen=200 #validation_data=[q_val_iterable, d_val_iterable, l_val_iterable],
                )

print('Test set results')
mp_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)
model_save_path = 'saved_models'
mp_model.save(os.path.join(model_save_path, 'my_mp_mpdel'))
