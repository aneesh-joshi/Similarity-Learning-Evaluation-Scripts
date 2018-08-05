import re
import os

class SickReader:
    """Reader object to provide training data"""
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

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def get_entailment_data(self):
        """Returns data in the format: SentA, SentB, Entailment Label"""
        return self.sentenceA, self.sentenceB, self.entailment_label

    def get_relatedness_data(self):
        """Returns data in the format: SentA, SentB, RelatednessScore"""
        return self.sentenceA, self.sentenceB, self.relatedness_score

    def get_entailment_label_dict(self):
        return self.entailment_label2index

if __name__ == '__main__':
    print(SickReader().get_relatedness_data()[1])
