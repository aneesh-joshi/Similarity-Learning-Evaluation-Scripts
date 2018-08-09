"""Utility script to convert the SQUAD 1.1 dataset into a QA dataset
(referred to as SQUAD-T in QA-Transfer paper)

Read more here : https://rajpurkar.github.io/SQuAD-explorer/

SQUAD 1.1 download link isn't available on the above website. However both 1.1 and 2.0 downloads
are automated in `data/get_data.py`

TODO : check if it works with SQUAD 2.0 (no reason it shouldn't)
       Even if it does work with 2.0, it won't be of value as a QA dataset since

       > "SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 new, 
       unanswerable questions written adversarially by crowdworkers to look 
       similar to answerable ones. To do well on SQuAD2.0, systems must not only
       answer questions when possible, but also determine when no answer is supported
       by the paragraph and abstain from answering"

The SQUAD-T dataset or as I like to call it : SQUAD-T-QA dataset can be saved in several formats
but I prefer to save it in the WikiQA format since I can then read it from the
data_readers.wiki_reader.WikiReaderIterable class.

Usage
-----
$ python squad2QA.py --squad_path path_to_squad_train.json

Example:
python squad2QA.py --squad_path ../data/train-v1.1.json


How it works
------------
SQUAD dataset is for span level QA
    passage: "Bob went to the doctor. He wasn't feeling too good"
    question: "Who went to the doctor"
    span: "*Bob* went to the doctor. He wasn't feeling too good"

This can be remodelled such that:
question : "Who went to the doctor"
doc1 : "Bob went to the doctor."
relevance : 1 (True)

question : "Who went to the doctor"
doc2 : "He wasn't feeling too good"
relevance : 0 (False)

Effectively, we get a really big good dataset in the QA domain. The converted file is almost 110 MB.
"""

import json
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--squad_path', required=True, help='path to the squad json')

    args = parser.parse_args()

    squad_file = args.squad_path

    qa_file_name = 'SQUAD-T-QA+.tsv'
    with open(qa_file_name, 'w', encoding='utf-8') as f:
        j = json.load(open(squad_file, encoding='utf-8'))

        header = 'QuestionID    Question    DocumentID  DocumentTitle   SentenceID  Sentence    Label'
        f.write(header + '\n')
        rows = []
        question_id = 0
        ocntr, icntr = 0, 0

        for data in j['data']:
            for paragraph in data['paragraphs']:
                sents = []
                sent_lens = []
                for sent in sent_tokenize(paragraph['context']):
                    sents.append(sent)
                    if len(sent_lens) == 0:
                        sent_lens.append(len(sent))
                    else:
                        sent_lens.append(len(sent) + sent_lens[-1])
                for qas in paragraph['qas']:
                    q = qas['question']
                    for answer in qas['answers']:
                        answer_start = answer['answer_start']
                        for i in range(len(sent_lens)):
                            if answer_start - sent_lens[i] <= 0:
                                relevant_ans_index = i
                                break
                    ocntr += 1
                    question_id += 1
                    inner_doc_id = 0
                    for i, s in enumerate(sents):
                        inner_doc_id += 1
                        icntr += 1
                        if i == relevant_ans_index:
                           f.write('Q'+str(question_id) + '\t' + q + '\t' + 'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(1)+ '\n')
                        else:
                            f.write('Q'+str(question_id) + '\t' + q + '\t' +\
                              'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(0)+ '\n')

    print('write complete. File saved as %s' % qa_file_name)
