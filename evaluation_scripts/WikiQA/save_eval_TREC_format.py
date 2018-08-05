import sys
sys.path.append('../..')
import os
import csv
import re
import gensim.downloader as api
from gensim.utils import simple_preprocess
import numpy as np
from sl_eval.models import DRMM_TKS
from sl_eval.models import MatchPyramid
import argparse

"""This script will save the TREC format of the predicions of a word2vec model and a drmm tks model

Usage
-----
$ python save_eval_TREC_format.py --wikiqa_path experimental_data/WikiQACorpus/WikiQA-test.tsv --model_path path/to/model --model_type mp

Please refer to http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system
to get a complete understanding of trec

Here's the short version:
TREC stands for Text REtrieval Conference going on since 2000.
They have a standard sets of tasks and an evaluation script for measuring metrics.

This evaluation has become a standard for evaluation of IR systems.

As the above tutorial link will tell, you will have to download the TREC evaluation code from https://trec.nist.gov/trec_eval/
run `make` in the folder and it will produce the `trec_eval` binary.

The binary requires 2 inputs:
1. qrels
    It is the query relevance file which will hold the correct answers.
2. pred file
    It is the predicition file which will have the predictions made by your model

$ ./trec_eval qrels pred

The above will provide some metrics. You can specify metrics like:
$ ./trec_eval -m map -m ndcg_cut.1,3,5,10,20 qrels pred

The above will provide MAP value and nDCG at cutoffs of 1, 3, 5, 10 and 20


This script will go through the WikiQA data and provide the qrels
It will then take the w2v model and your trained model and provide the predicted scores.
Provide the qrels and pred to your `trec_eval` binary and run it to get the evaluations.
"""

class MyWikiIterable:

    def __init__(self, fpath):
        # self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        # self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)
        self.to_print = ""

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def  get_stuff(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        ANSWER_ID_INDEX = 4
        LABEL_INDEX = 6

        document_group = []
        label_group = []

        n_relevant_docs = 0
        n_filtered_docs = 0

        query_ids = []
        query_id_group = []
        doc_ids = []
        doc_id_group = []
        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))


                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                        doc_id_group.append(doc_ids)
                    else:
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []
                    doc_ids = []
            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])


                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    doc_id_group.append(doc_ids)
                    query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0
        return queries, docs, labels, query_ids, doc_id_group


def save_qrels(fname):
    """Saves the WikiQA data `Truth Data`. This remains the same regardless of which model you use.
    qrels : query relevance

    Format
    ------
    <query_id>\t<0>\t<document_id>\t<relevance>

    Note: parameter <0> is ignored by the model

    Example
    -------
    Q1  0   D1-0    0
    Q1  0   D1-1    0
    Q1  0   D1-2    0
    Q1  0   D1-3    1
    Q1  0   D1-4    0
    Q16 0   D16-0   1
    Q16 0   D16-1   0
    Q16 0   D16-2   0
    Q16 0   D16-3   0
    Q16 0   D16-4   0

    Parameters
    ----------
    fname : str
        File where the qrels should be saved

    """
    with open(fname, 'w') as f:
        for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
            for d, l, d_id in zip(doc, labels, d_ids):
                f.write(q_id + '\t' +  '0' + '\t' +  str(d_id) + '\t' + str(l) + '\n')
    print("qrels done. Saved as %s" % fname)

def save_model_pred(fname, similarity_fn):
    """Goes through all the queries and docs, gets their Similarity score as per the `similarity_fn`
    and saves it in the TREC format

    Format
    ------
    <query_id>\t<Q0>\t<document_id>\t<rank>\t<model_score>\t<STANDARD>

    Note: parameters <Q0>, <rank> and <STANDARD> are ignored by the model and can be kept as anything
    I have chose 99 as the rank. It has no meaning.

    Example
    -------
    Q1  Q0  D1-0    99  0.64426434  STANDARD
    Q1  Q0  D1-1    99  0.26972288  STANDARD
    Q1  Q0  D1-2    99  0.6259719   STANDARD
    Q1  Q0  D1-3    99  0.8891963   STANDARD
    Q1  Q0  D1-4    99  1.7347554   STANDARD
    Q16 Q0  D16-0   99  1.1078827   STANDARD
    Q16 Q0  D16-1   99  0.22940424  STANDARD
    Q16 Q0  D16-2   99  1.7198141   STANDARD
    Q16 Q0  D16-3   99  1.7576259   STANDARD
    Q16 Q0  D16-4   99  1.548423    STANDARD

    Parameters
    ----------
    fname : str
        File where the qrels should be saved

    similarity_fn : function
        Parameters
            - query : list of str
            - doc : list of str
        Returns
            - similarity_score : float
    """
    with open(fname, 'w') as f:
        for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
            for d, l, d_id in zip(doc, labels, d_ids):
                my_score = str(similarity_fn(q,d))
                f.write(q_id + '\t' + 'Q0' + '\t' + str(d_id) + '\t' + '99' + '\t' + my_score + '\t' + 'STANDARD' + '\n')
    print("Prediction done. Saved as %s" % fname)

def w2v_similarity_fn(q, d):
    """Similarity Function for Word2Vec

    Parameters
    ----------
    query : list of str
    doc : list of str

    Returns
    -------
    similarity_score : float
    """
    def sent2vec(sent):
        if len(sent)==0:
            print('length is 0, Returning random')
            return np.random.random((dim_size,))

        vec = []
        for word in sent:
            if word in kv_model:
                vec.append(kv_model[word])

        if len(vec) == 0:
            print('No words in vocab, Returning random')
            return np.random.random((kv_model.vector_size,))

        vec = np.array(vec)

        return np.mean(vec, axis=0)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)* np.linalg.norm(vec2))
    return cosine_similarity(sent2vec(q),sent2vec(d))

def dtks_similarity_fn(q, d):
    """Similarity Function for DRMM TKS

    Parameters
    ----------
    query : list of str
    doc : list of str

    Returns
    -------
    similarity_score : float
    """
    return dtks_model.predict([q], [[d]])[0][0]

def mp_similarity_fn(q, d):
    """Similarity Function for DRMM TKS

    Parameters
    ----------
    query : list of str
    doc : list of str

    Returns
    -------
    similarity_score : float
    """
    return mp_model.predict([q], [[d]])[0][0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save the TREC format results')

    parser.add_argument('--wikiqa_path', help='Path to WikiQA .tsv file')
    parser.add_argument('--model_path', help='Path to DRMM_TKS or other such model')
    parser.add_argument('--model_type')
    parser.add_argument('--do_w2v', default=False)
    parser.add_argument('--w2v_dim')

    args = parser.parse_args()
    wikiqa_path = args.wikiqa_path
    model_path = args.model_path
    model_type = args.model_type
    do_w2v = args.do_w2v
    w2v_dim = args.w2v_dim

    queries, doc_group, label_group, query_ids, doc_id_group = MyWikiIterable(wikiqa_path).get_stuff()

    print('queries :', len(queries))
    print('doc_groups : ', len(doc_group))
    print('label_groups : ', len(label_group))
    print('query_ids : ', len(query_ids))
    print('doc_id_groups : ', len(doc_id_group))

    # Get the qrels, which will be the same for every model
    save_qrels('qrels')
    if model_type == 'dtks':
        dtks_model = DRMM_TKS.load(model_path)
        save_model_pred(str(model_path) + 'pred', dtks_similarity_fn)
    elif model_type == 'mp':
        mp_model = MatchPyramid.load(model_path)
        save_model_pred(str(model_path) + 'pred', mp_similarity_fn)
#    elif model_type == 'bimpm':
        

