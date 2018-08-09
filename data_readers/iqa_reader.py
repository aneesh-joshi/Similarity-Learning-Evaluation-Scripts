import pickle
import random
import os
import warnings
from sklearn.utils import shuffle

class IQAReader:
    """Class to read the InsuranceQA dataset and provide train, test and dev samples according
    to specification
    
    About InsuranceQA
    -----------------
    This dataset was originally released here : TODO
    It has several different formats but at its base there is (for one data point):
    - a question
    - one or more correct answers
    - a pool of all answers (correct and incorrect)

    It doesn't have a simple format like question - document - relevance
    So, we'll have to convert it to the QA format.
    That basically involves taking a question, its correct answer and marking it as relevant.
    Then for the remaining number of answers(however big you want the batch size to be), we pick
    (batch_size - current_size) a=

    The original repo has several files and is *very* confusing.
    Luckily, there is a converted version of it here : TODO

    This class directly making use of that version and provides it in the QA format

    Parameters
    ----------
    folder_path : str
    path to the folder cloned from TODO

    """ 

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.vocab = self._get_pickle('vocabulary')
        self.answer_pool = self._get_pickle('answers')
        self.num_answers = len(self.answer_pool)


    def _get_pickle(self, filename):
        """Unpickles and loads a file"""
        return pickle.load(open(os.path.join(self.folder_path, filename), 'rb'))

    def _translate_sent(self, sent):
        """Translates a sentence from the index format to the string format

        Note: InsuranceQA for some reason provides words as indices along with a conversion dict
        So, the dataset looks like : [12, 345, 23, ...]
        with {12 : 'hello', 345: 'world', ...}

        This function translates a given indexed sentence

        Parameters
        ----------
        sent : list of int
            The sentence to be translated

        Returns
        -------
        list of str
        """
        return [self.vocab[word] for word in sent]

    def _get_answer(self, answer_id):
        """Gets the sentence from a pool of answers using the answer_id and translates it using the
        translation dict

        Parameters
        ----------
        answer_id : int
            The id of the answer which needs translation.

        Returns
        -------
        list of str
        """
        return self._translate_sent(self.answer_pool[answer_id])

    def _get_pool_answer(self, correct_answer_ids):
        """Gets one random answer from the pool of answers which isn't correct_answer_ids(can be more than one)

        Note: Since there is a pool of answers, we will pick a negative answer at random. However, it cannot be
        the same aas the correct answer. If you get the correct answer by chance, re-sample from the pool

        Parameters
        ----------
        correct_answer_ids : list of int
            The ids of the correct answers

        Returns
        -------
        list of str
        """
        answer_id = random.randint(1, self.num_answers)
        while (set([answer_id]) <= set(correct_answer_ids)) == True:
            answer_id = random.randint(1, self.num_answers)
        return self._get_answer(answer_id)


    def get_train_data(self, batch_size):
        """Gets the training data in batches of `batch_size`

        Initially, we take a question, its correct answer and label it as relevant.
        Then, we check if the size if filling the batch (it shouldn't)

        For the reamining empty slots in the batch, we sample an answer from the pool
        of answers which isn't the correct answer and label it as irrelevant.
        Lastly, we shuffle the batches and return it.
        
        Beware: Due to the stochastic sampling nature of this dataset, we can get
        different datasets from run to run.

        Parameters
        ----------
        batch_size : int
            The size of the batches of training data
        """
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
        """Gets the testing data in a format which allows evalution in the
        TREC format needed by the calling script. You can still use it for
        general purposes.

        Beware: Due to the stochastic sampling nature of this dataset, we can get
        different datasets from run to run.

        Parameters
        ----------
        split : {'test1', 'test2'}
            InsuraceQA provides these two test sets.
        batch_size : int
            The size of the batches of training data
        """
        testx = self._get_pickle(split)
        batch_a, batch_l = [], []
        questions, answers, labels, question_ids, doc_ids = [], [], [], [], []
        question_ids, batch_doc_ids = [], []

        for i, item in enumerate(testx):
            questions.append(self._translate_sent(item['question']))
            question_ids.append('Q-{}'.format(i))
            for j, answer_id in enumerate(item['good']):
                batch_a.append(self._get_answer(answer_id))
                batch_doc_ids.append('D{}-{}'.format(i, j))
                batch_l.append(1)
            while(len(batch_a) < batch_size):
                j += 1
                batch_a.append(self._get_pool_answer(item['good']))
                batch_doc_ids.append('D{}-{}'.format(i, j))
                batch_l.append(0)
            batch_a, batch_doc_ids, batch_l = shuffle(batch_a, batch_doc_ids, batch_l)
            doc_ids.append(batch_doc_ids)
            answers.append(batch_a)
            labels.append(batch_l)
            batch_a, batch_l, batch_doc_ids = [], [], []

        return [questions, answers, labels, question_ids, doc_ids]
