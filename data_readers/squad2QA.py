import json
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess


file_name = 'SQUAD-T-QA.tsv'

with open(file_name, 'w') as f:
	j = json.load(open('dev-v1.1.json'))

	header = 'QuestionID	Question	DocumentID	DocumentTitle	SentenceID	Sentence	Label'
	f.write(header + '\n')
	rows = []
	temp_str = ''

	question_id = 0


	ocntr, icntr = 0, 0
	for data in j['data']:
		for paragraph in data['paragraphs']:
			sents = []
			sent_lens = []
			# print(paragraph['context'])
			# print('----------------------')
			for sent in sent_tokenize(paragraph['context']):
				# print(simple_preprocess(sent))
				sents.append(sent)
				if len(sent_lens) == 0:
					sent_lens.append(len(sent))
				else:
					sent_lens.append(len(sent) + sent_lens[-1])
			# print('*********************')
			# print(sent_lens)
			for qas in paragraph['qas']:
				# print(qas)
				q = qas['question']

				for answer in qas['answers']:
					answer_start = answer['answer_start']
					for i in range(len(sent_lens)):
						if answer_start - sent_lens[i] <= 0:
							relevant_ans_index = i
							break

				# print('*********************')
				ocntr += 1
				question_id += 1
				inner_doc_id = 0
				for i, s in enumerate(sents):
					inner_doc_id += 1
					icntr += 1
					if i == relevant_ans_index:
						f.write('Q'+str(question_id) + '\t' + q + '\t' +\
						  'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(1)+ '\n')
					else:
						f.write('Q'+str(question_id) + '\t' + q + '\t' +\
						  'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(0)+ '\n')
				# print('*********************')
	# print(ocntr, icntr)

print('write complete')