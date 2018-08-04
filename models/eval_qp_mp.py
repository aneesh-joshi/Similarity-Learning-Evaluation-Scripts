from gensim import downloader as api
from sklearn.utils import shuffle
from matchpyramid import MatchPyramid
import re

train_split = 0.8

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

train_q1, test_q1 = q1[:int(len(q1)*train_split)], q1[int(len(q1)*train_split):]
train_q2, test_q2 = q2[:int(len(q2)*train_split)], q2[int(len(q2)*train_split):]
train_duplicate, test_duplicate = duplicate[:int(len(duplicate)*train_split)], duplicate[int(len(duplicate)*train_split):]

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

kv_model = api.load('glove-wiki-gigaword-300')

batch_size = 100

mp_model = MatchPyramid(queries=train_q1, docs=train_q2, labels=train_duplicate, target_mode='classification', word_embedding=kv_model, epochs=40, text_maxlen=80, batch_size=batch_size, steps_per_epoch=323432//batch_size)
mp_model.evaluate_classification(test_q1, test_q2, test_duplicate)
