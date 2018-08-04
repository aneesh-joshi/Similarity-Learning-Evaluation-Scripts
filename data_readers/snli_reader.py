import json
import os
import re
from keras.utils.np_utils import to_categorical
class SnliReader:
	
	def __init__(self, filepath):
		self.filepath = filepath
		self.filename = {}
		self.filename['train'] = 'snli_1.0_train.jsonl'
		self.filename['dev'] = 'snli_1.0_dev.jsonl'
		self.filename['test'] = 'snli_1.0_test.jsonl'
		self.label2index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

	def get_data(self, split):
		x1, x2, labels, annotator_labels = [], [], [], []
		with open(os.path.join(self.filepath, self.filename[split]), 'r') as f:
			for line in f:
				line = json.loads(line)
				if line['gold_label'] == '-':
					# In the case of this unknown label, we will skip the whole datapoint
					continue
				x1.append(self._preprocess(line['sentence1']))
				x2.append(self._preprocess(line['sentence2']))
				labels.append(self.label2index[line['gold_label']])
				
				annotator_labels.append(line['annotator_labels'])

		return x1, x2, labels, annotator_labels

	def _preprocess(self, sent):
		return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

	def get_label2index(self):
		return self.label2index
