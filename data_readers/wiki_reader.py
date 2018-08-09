"""This file contains WikiReaderIterable and WikiReaderStatic for handling the WikiQA dataset

Use WikiReaderIterable when you want data in the format of query, docs, labels seperately
Example:
query_iterable = WikiReaderIterable('query', path_to_file)

Use WikiReaderStatic when you want a dump of the test data with the doc_ids and query_ids
It is useful for saving predictions in the TREC format

A datapoint in this dataset has a query, a document and thier relevance(0: irrelevant, 1: relevant)

Example data point:
QuestionID  Question    DocumentID  DocumentTitle   SentenceID  Sentence    Label
Q8  How are epithelial tissues joined together? D8  Tissue (biology)    D8-0    Cross section of sclerenchyma fibers in plant ground tissue 0

"""
import numpy as np
import re
import csv

class WikiReaderIterable:
    """Returns an iterable for the given `iter_type` after extracting from the WikiQA tsv

    Parameters
    ----------
    iter_type : {'query', 'doc', 'label'}
        The type of data point
    fpath : str
        Path to the .tsv file
    """

    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence
        Replace this function if you want to handle preprocessing differently

        Parameters
        ----------
        sent : str
        """
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def __iter__(self):
        # Defining some consants for .tsv reading
        # These refer to the column indexes of certain data
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        LABEL_INDEX = 6

        # We will be grouping all documents and labels which belong to one question into
        # one group. This helps in getting MAP scores.
        document_group = []
        label_group = []

        # We keep count of number of documents so we can remove those question-doc pairs
        # which do not have even one relevant document
        n_relevant_docs = 0
        n_filtered_docs = 0

        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                # If the question id index doesn't change
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                    else:
                        # Filter out a question if it doesn't have a single relevant document
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []

            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    # Return the index of the doc requested
                    yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0


class WikiReaderStatic:
    """Returns test data with some meta data needed for returning data in the TREC format

    Parameters
    ----------
    fpath : str
        Path to the .tsv file

    Call the `get_data` function to get the test_data
    """
    def __init__(self, fpath):
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence
        Replace this function if you want to handle preprocessing differently

        Parameters
        ----------
        sent: str
        """
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def  get_data(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        ANSWER_ID_INDEX = 4
        LABEL_INDEX = 6

        # We will be grouping all documents and labels which belong to one question into
        # one group. This helps in getting MAP scores.
        document_group = []
        label_group = []

        # We keep count of number of documents so we can remove those question-doc pairs
        # which do not have even one relevant document
        n_relevant_docs = 0
        n_filtered_docs = 0

        query_ids = []
        query_id_group = []
        doc_ids = []
        doc_id_group = []
        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])

                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))


                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                        doc_id_group.append(doc_ids)
                    else:
                        # Filter out a question if it doesn't have a single relevant document
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []
                    doc_ids = []
            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                doc_ids.append(self.data_rows[i][ANSWER_ID_INDEX])


                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    doc_id_group.append(doc_ids)
                    query_ids.append(self.data_rows[i][QUESTION_ID_INDEX])
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0
        return [queries, docs, labels, query_ids, doc_id_group]
