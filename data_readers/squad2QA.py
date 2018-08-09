"""Utility script to convert the SQUAD 1.1 dataset into a QA dataset
(referred to as SQUAD-T in QA-Transfer paper)

TODO : check if it works with SQUAD 2.0

Caution : This script has a problem when dealing with chemistry equations. For some reason, the way the reading is done,
the subscript is taken as an en

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
    with open(qa_file_name, 'w') as f:
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
                            f.write('Q'+str(question_id) + '\t' + q + '\t' +\
                              'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(1)+ '\n')
                        else:
                            f.write('Q'+str(question_id) + '\t' + q + '\t' +\
                              'D'+str(question_id) + '\t' + 'TempDocTitle' + '\t' + 'D'+str(question_id)+'-'+str(inner_doc_id)+'\t'+s+'\t'+ str(0)+ '\n')

    print('write complete')