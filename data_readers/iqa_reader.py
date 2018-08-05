import pickle
import random
import os
import warnings
from sklearn.utils import shuffle

class IQAReader:
    """Class to read the InsuranceQA dataset and provide train, test and dev samples according
    to specification""" 

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.vocab = self._get_pickle('vocabulary')
        self.answer_pool = self._get_pickle('answers')
        self.num_answers = len(self.answer_pool)


    def _get_pickle(self, filename):
        return pickle.load(open(os.path.join(self.folder_path, filename), 'rb'))

    def _translate_sent(self, sent):
        return [self.vocab[word] for word in sent]

    def _get_answer(self, answer_id):
        return self._translate_sent(self.answer_pool[answer_id])

    def _get_pool_answer(self, correct_answer_ids):
        answer_id = random.randint(1, self.num_answers)
        while (set([answer_id]) <= set(correct_answer_ids)) == True:
            answer_id = random.randint(1, self.num_answers)
        return self._get_answer(answer_id)


    def get_train_data(self, batch_size):
        """Gets the training data"""
        train = self._get_pickle('train')
        batch_a, batch_l = [], []
        questions, answers, labels = [], [], []

        for item in train:
            for answer_id in item['answers']:
                batch_a.append(self._get_answer(answer_id))
                batch_l.append(1)
            if len(batch_a) > batch_size:
                print(item['answers'])
                raise ValueError(
                            "The number of correct answers: %d is bigger than the batch_size: %d"
                            "Consider increasing the batch_size" % (len(batch_a), batch_size)
                        )
            while(len(batch_a) < batch_size):
                batch_a.append(self._get_pool_answer(item['answers']))
                batch_l.append(0)

            questions.append(self._translate_sent(item['question']))
            answers.append(batch_a)
            labels.append(batch_l)

            batch_a, batch_l = [], []

        return questions, answers, labels

    def get_test_data(self, split, batch_size):
        """Gets the training data

        split : {'test1', 'test2'}
        """
        testx = self._get_pickle(split)
        batch_a, batch_l = [], []
        questions, answers, labels = [], [], []

        for item in testx:
            questions.append(self._translate_sent(item['question']))
            for answer_id in item['good']:
                batch_a.append(self._get_answer(answer_id))
                batch_l.append(1)
            while(len(batch_a) < batch_size):
                batch_a.append(self._get_pool_answer(item['good']))
                batch_l.append(0)
            answers.append(batch_a)
            labels.append(batch_l)
            batch_a, batch_l = [], []

        return questions, answers, labels

if __name__ == '__main__':
    iqa_reader = InsuranceQAReader(pool_size=32)
    print(iqa_reader.get_train_data()[2])


    # questions, docs, labels = {}, {}, {}


    # for name in ['dev', 'test1', 'test2']:

    #   dev = _get_pickle(name)
    #   print('There are %d questions in %s' % (len(dev), name))

    #   dev_questions = []
    #   dev_answers = []
    #   dev_labels = []

    #   for data_item in dev:
    #       dev_questions.append(self._translate_sent(data_item['question']))
    #       dev_batch_answer, dev_batch_label = [], []
    #       dev_batch_answer.append(self._get_answer(data_item['good'][0]))
    #       dev_batch_label.append(1)

    #       for bad_answer in data_item['bad']:
    #           dev_batch_answer.append(self._get_answer(bad_answer))
    #           dev_batch_label.append(0)
    #       dev_answers.append(dev_batch_answer)
    #       dev_labels.append(dev_batch_label)
    #       dev_batch_answer, dev_batch_label = [], []

    #   questions[name] = dev_questions
    #   docs[name] = dev_answers
    #   labels[name] = dev_labels

    #   # for q, doc, label in zip(dev_questions, dev_answers, dev_labels):
    #   #   print(len(q), len(doc), len(label))
    #   #   print('=================================================\n\n\n\n\n\n\n')
