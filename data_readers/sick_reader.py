import re
import os

class SickReader:
    """Reader object to provide training data from the SICK dataset
    More details can be found here : http://clic.cimec.unitn.it/composes/sick.html

    The dataset contains:
    - two sentences, SentenceA and SentenceB
    - thier relation to each other (entailment, contradiction, neutral)
    - their relatedness score (sentA and sentB have a relatedness score of 3.4)
    
    Example row in the dataset:
    pair_ID sentence_A  sentence_B  entailment_label    relatedness_score   entailment_AB   entailment_BA   sentence_A_original sentence_B_original sentence_A_dataset  sentence_B_dataset  SemEval_set
    1   A group of kids is playing in a yard and an old man is standing in the background   A group of boys in a yard is playing and a man is standing in the background    NEUTRAL 4.5 A_neutral_B B_neutral_A A group of children playing in a yard, a man in the background. A group of children playing in a yard, a man in the background. FLICKR  FLICKR  TRAIN

    You can get the entailment data with labels with `get_entailment_data`
    You can get the relatedness data with labels with `get_relatedness_data`


    Parameters
    ----------
    filepath : str
        path to folder with SICK.txt
    preprocess_fn : function
        function to preprocess sentences.
        If None, will use `self._preprocess_fn`

    """
    def __init__(self, filepath, preprocess_fn=None):
        if preprocess_fn != None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = self._preprocess_fn
        SENTA_INDEX, SENTB_INDEX, ENTAILMENT_INDEX, RELATEDNESS_INDEX = 1, 2, 3, 4
        SPLIT_INDEX = 11
        self.sentenceA, self.sentenceB, self.entailment_label, self.relatedness_score = {}, {}, {}, {}

        for s in [self.sentenceA, self.sentenceB, self.entailment_label, self.relatedness_score]:
            s['TRAIN'], s['TEST'], s['TRIAL'] = [], [], []


        self.entailment_label2index = {'CONTRADICTION': 0, 'ENTAILMENT': 1, 'NEUTRAL': 2}
        splits = []
        with open(os.path.join(filepath, 'SICK.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # to skip the header
                line = line[:-1]
                split_line = line.split('\t')
                self.sentenceA[split_line[SPLIT_INDEX]].append(self.preprocess_fn(split_line[SENTA_INDEX]))
                self.sentenceB[split_line[SPLIT_INDEX]].append(self.preprocess_fn(split_line[SENTB_INDEX]))
                self.entailment_label[split_line[SPLIT_INDEX]].append(self.entailment_label2index[split_line[ENTAILMENT_INDEX]])
                self.relatedness_score[split_line[SPLIT_INDEX]].append(float(split_line[RELATEDNESS_INDEX])) 

    def _preprocess_fn(self, sent):
        """Utility function to lower, strip and tokenize each sentence(on spaces)
        Replace this function if you want to handle preprocessing differently
        
        Parameters
        ----------
        sent : str
            The string sentence
        """
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def get_entailment_data(self):
        """Returns data in the format: SentA, SentB, Entailment Label

        where each of them is a dict which contains the train, test and trial data in 
        'TRAIN', 'TEST' and 'TRIAL' keys respectively"""
        return self.sentenceA, self.sentenceB, self.entailment_label

    def get_relatedness_data(self):
        """Returns data in the format: SentA, SentB, RelatednessScore
        where each of them is a dict which contains the train, test and trial data in 
        'TRAIN', 'TEST' and 'TRIAL' keys respectively"""
        return self.sentenceA, self.sentenceB, self.relatedness_score

    def get_entailment_label_dict(self):
        """Returns the mapping from int to label(str)
        {'CONTRADICTION': 0, 'ENTAILMENT': 1, 'NEUTRAL': 2}"""
        return self.entailment_label2index

