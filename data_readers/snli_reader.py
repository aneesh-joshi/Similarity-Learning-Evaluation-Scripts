import json
import os
import re
from keras.utils.np_utils import to_categorical

class SnliReader:
	"""Reader for the SNLI dataset
	More details can be found here : https://nlp.stanford.edu/projects/snli/

	Each data point contains 2 sentences and their label('contradiction', 'entailment', 'neutral')
	Additionally, it also provides annotator labels which has a range of labels given by the annotators. We will mostly ignore this.

	Example datapoint:
	gold_label	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	captionID	pairID	label1	label2	label3	label4	label5
	neutral	( ( Two women ) ( ( are ( embracing ( while ( holding ( to ( go packages ) ) ) ) ) ) . ) )	( ( The sisters ) ( ( are ( ( hugging goodbye ) ( while ( holding ( to ( ( go packages ) ( after ( just ( eating lunch ) ) ) ) ) ) ) ) ) . ) )	(ROOT (S (NP (CD Two) (NNS women)) (VP (VBP are) (VP (VBG embracing) (SBAR (IN while) (S (NP (VBG holding)) (VP (TO to) (VP (VB go) (NP (NNS packages)))))))) (. .)))	(ROOT (S (NP (DT The) (NNS sisters)) (VP (VBP are) (VP (VBG hugging) (NP (UH goodbye)) (PP (IN while) (S (VP (VBG holding) (S (VP (TO to) (VP (VB go) (NP (NNS packages)) (PP (IN after) (S (ADVP (RB just)) (VP (VBG eating) (NP (NN lunch))))))))))))) (. .)))	Two women are embracing while holding to go packages.	The sisters are hugging goodbye while holding to go packages after just eating lunch.	4705552913.jpg#2	4705552913.jpg#2r1n	neutral	entailment	neutral	neutral	neutral

	Parameters
	----------
	filepath : str
		path to the folder with the snli data

	"""
	
	def __init__(self, filepath):
		self.filepath = filepath
		self.filename = {}
		self.filename['train'] = 'snli_1.0_train.jsonl'
		self.filename['dev'] = 'snli_1.0_dev.jsonl'
		self.filename['test'] = 'snli_1.0_test.jsonl'
		self.label2index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

	def get_data(self, split):
		"""Returnd the data for the given split

		Parameters
		----------
		split : {'train', 'test', 'dev'}
			The split of the data

		Returns
		-------
		sentA_datalist, sentB_datalist, lablels, annotator_labels
		"""
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
		"""lower, strip and split the string and remove unnecessaey characters

		Parameters
		----------
		sent : str
			The sentence to be preprocessed
		"""
		return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

	def get_label2index(self):
		"""Returns the label2index dict"""
		return self.label2index
