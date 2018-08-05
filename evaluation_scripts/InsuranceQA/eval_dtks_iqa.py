import sys
import os
sys.path.append('../..')
from sklearn.utils import shuffle
from data_readers import IQAReader
import gensim.downloader as api
from sl_eval.models import DRMM_TKS

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

iqa_folder_path = os.path.join('..', '..', 'data', 'insurance_qa_python')
iqa_reader = IQAReader(iqa_folder_path)


# PARAMETERS ---------------------------------------------------------
train_batch_size = 50
test_batch_size = 10
word_embedding_len = 300
batch_size = 50
text_maxlen = 300
n_epochs = 5 
# --------------------------------------------------------------------

train_q, train_d, train_l = iqa_reader.get_train_data(batch_size=train_batch_size)

train_q, train_d, train_l = shuffle(train_q, train_d, train_l)

kv_model = api.load('glove-wiki-gigaword-' + str(word_embedding_len))
steps_per_epoch = len(train_q)//batch_size
dtks_model = DRMM_TKS(queries=train_q, docs=train_d, labels=train_l, target_mode='ranking',
                      word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch)

queries, doc_group, label_group, query_ids, doc_id_group = iqa_reader.get_test_data('test1', batch_size=test_batch_size)
save_qrels('qrels1_dtks')
save_model_pred('pred1_dtks_iqa', dtks_similarity_fn)

queries, doc_group, label_group, query_ids, doc_id_group = iqa_reader.get_test_data('test2', batch_size=test_batch_size)
save_qrels('qrels2_dtks')
save_model_pred('pred2_dtks_iqa', dtks_similarity_fn)


# for q, doc, label, qid, dids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
#   for d, l, did in zip(doc, label, dids):
#       print(q, d, l, qid, did)
