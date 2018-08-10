"""This script evaluates the DRMM_TKS, MatchPyramid and BiDAF-T model on Quora Duplicate Questions Task"""

import sys
sys.path.append('..')

import numpy as np
from gensim import downloader as api
from sklearn.utils import shuffle
from sl_eval.models.matchpyramid import MatchPyramid
from sl_eval.models.drmm_tks import DRMM_TKS
import re

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
            return np.random.random((kv_model.vector_size,))

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

if __name__ == '__main__':

	train_split = 0.8
	num_samples = 323432
	n_word_embedding_dims = 300

	qqp = api.load('quora-duplicate-questions')

	def preprocess(sent):
		return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

	sent_len = []

	q1, q2, duplicate = [], [], []
	for row in qqp:
		sent_len.append(len(row['question1']))
		sent_len.append(len(row['question2']))
		q1.append(preprocess(row['question1']))
		q2.append(preprocess(row['question2']))
		duplicate.append(int(row['is_duplicate']))

	print(sum(sent_len)/len(sent_len))

	print('Number of question pairs', len(q1))
	print('Number of duplicates', sum(duplicate))
	print('% duplicates', 100.*sum(duplicate)/len(q1))
	print('-----------------------------------------')

	q1, q2, duplicate = shuffle(q1, q2, duplicate)

	train_q1, test_q1 = q1[:int(len(q1) * train_split)], q1[int(len(q1) * train_split):]
	train_q2, test_q2 = q2[:int(len(q2) * train_split)], q2[int(len(q2) * train_split):]
	train_duplicate, test_duplicate = duplicate[:int(len(duplicate) * train_split)], duplicate[int(len(duplicate) * train_split):]

	assert len(train_q1) == len(train_duplicate)
	assert len(test_q2) == len(test_duplicate)


	print('Number of question pairs in train', len(train_q1))
	print('Number of duplicates in train', sum(train_duplicate))
	print('%% duplicates', 100.*sum(train_duplicate)/len(train_q1))
	print('-----------------------------------------')

	print('Number of question pairs in test', len(test_q1))
	print('Number of duplicates in test', sum(test_duplicate))
	print('%% duplicates', 100.*sum(test_duplicate)/len(test_q1))
	print('-----------------------------------------')


	kv_model = api.load('glove-wiki-gigaword-' + str(n_word_embedding_dims))

	print('Getting word2vec baseline')
	num_correct, num_total = 0, 0
	for q1, q2, d in zip(test_q1, test_q2, test_duplicate):
		if w2v_similarity_fn(q1, q2) >= 0 and d == 1 or w2v_similarity_fn(q1, q2) < 0 and d == 0:
			num_correct += 1
		num_total += 1
	print('Word2Vec baseline accuracy is %f' % (100. * num_correct / num_total))


	n_epochs = 10
	batch_size = 100
	text_maxlen = 80
	mp_model = MatchPyramid(queries=train_q1, docs=train_q2, labels=train_duplicate, target_mode='classification',
							word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size,
							steps_per_epoch=num_samples//batch_size)
	num_correct, num_total, accuracy = mp_model.evaluate_classification(test_q1, test_q2, test_duplicate, batch_size=20)
	print('Results on MatchPyramid with Quora Duplicate Questions dataset')
	print('Accuracy = %.2f' % accuracy*100)
	print('Predicted %d correct out of a totol of %d' % (num_correct, num_total))

	n_epochs = 10 
	batch_size = 10
	text_maxlen = 200
	dtks_model = DRMM_TKS(queries=train_q1, docs=train_q2, labels=train_duplicate, target_mode='classification',
	                     word_embedding=kv_model, epochs=n_epochs, text_maxlen=text_maxlen, batch_size=batch_size,
	                     steps_per_epoch=num_samples//batch_size)
	num_correct, num_total, accuracy = dtks_model.evaluate_classification(test_q1, test_q2, test_duplicate, batch_size=20)
	print('Results on MatchPyramid with Quora Duplicate Questions dataset')
	print('Accuracy = %.2f' % accuracy*100)
	print('Predicted %d correct out of a totol of %d' % (num_correct, num_total))
