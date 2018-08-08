import sys
sys.path.append('../..')
import sys
import os

from sl_eval.models import MatchPyramid, DRMM_TKS
from data_readers import WikiReaderIterable, WikiReaderStatic
import gensim.downloader as api

def save_qrels(test_data, fname):
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
    queries, doc_group, label_group, query_ids, doc_id_group = test_data
    with open(fname, 'w') as f:
        for q, doc, labels, q_id, d_ids in zip(queries, doc_group, label_group, query_ids, doc_id_group):
            for d, l, d_id in zip(doc, labels, d_ids):
                f.write(q_id + '\t' +  '0' + '\t' +  str(d_id) + '\t' + str(l) + '\n')
    print("qrels done. Saved as %s" % fname)

def save_model_pred(test_data, fname, similarity_fn):
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
    queries, doc_group, label_group, query_ids, doc_id_group = test_data
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
	wikiqa_folder = os.path.join('..', '..', 'data', 'WikiQACorpus')

	q_iterable = WikiReaderIterable('query', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))
	d_iterable = WikiReaderIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))
	l_iterable = WikiReaderIterable('label', os.path.join(wikiqa_folder, 'WikiQA-train.tsv'))

	q_val_iterable = WikiReaderIterable('query', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))
	d_val_iterable = WikiReaderIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))
	l_val_iterable = WikiReaderIterable('label', os.path.join(wikiqa_folder, 'WikiQA-dev.tsv'))

	q_test_iterable = WikiReaderIterable('query', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))
	d_test_iterable = WikiReaderIterable('doc', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))
	l_test_iterable = WikiReaderIterable('label', os.path.join(wikiqa_folder, 'WikiQA-test.tsv'))

	test_data = WikiReaderStatic(os.path.join(wikiqa_folder, 'WikiQA-test.tsv')).get_data()

	num_samples = 9000
	num_embedding_dims = 300
	qrels_save_path = 'qrels_wikiqa'
	mp_pred_save_path = 'pred_mp_wikiqa'
	dtks_pred_save_path = 'pred_dtks_wikiqa'
	
	print('Saving qrels for WikiQA test data')
	save_qrels(test_data, qrels_save_path)

	kv_model = api.load('glove-wiki-gigaword-' + str(num_embedding_dims))

	n_epochs = 2
	batch_size = 10
	steps_per_epoch = num_samples // batch_size
	#steps_per_epoch = 1

	# Train the model
	mp_model = MatchPyramid(
	                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model,
	                    epochs=n_epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, text_maxlen=200
	                )

	print('Test set results')
	mp_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)

	print('Saving prediction on test data in TREC format')
	save_model_pred(test_data, mp_pred_save_path, mp_similarity_fn)

	batch_size = 10
	steps_per_epoch = num_samples // batch_size

	# Train the model
	drmm_tks_model = DRMM_TKS(
	                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=3,
	                    topk=20, steps_per_epoch=steps_per_epoch, batch_size=batch_size
	                )

	print('Test set results')
	drmm_tks_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)

	print('Saving prediction on test data in TREC format')
	save_model_pred(test_data, dtks_pred_save_path, dtks_similarity_fn)

