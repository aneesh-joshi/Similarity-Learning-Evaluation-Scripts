from gensim.similarity_learning import WikiQA_DRMM_TKS_Extractor
from gensim.similarity_learning.preprocessing import ListGenerator
from gensim.similarity_learning.models import DRMM_TKS
from gensim.similarity_learning import rank_hinge_loss
from gensim.similarity_learning import ValidationCallback
from gensim.similarity_learning import mapk, mean_ndcg
import numpy as np
import pandas as pd
import argparse
import os
import logging

logger = logging.getLogger(__name__)

"""
Script to evaluate an untrained drmm tks model
"""

if __name__ == '__main__':

    logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    datapath = os.path.join("..", "data", "WikiQACorpus")
    word_embedding_path = os.path.join("..", "evaluation_scripts", "glove.6B.50d.txt")

    train_file_path = os.path.join(datapath, 'WikiQA-train.tsv')
    test_file_path = os.path.join(datapath, 'WikiQA-test.tsv')
    dev_file_path = os.path.join(datapath, 'WikiQA-dev.tsv')

    mets = [0, 0, 0, 0, 0, 0]
    map_val = 0
    n1 = n3 = n5 = n10 = n20 = 0
    n_iters = 20
    for i in range(n_iters):
        print("\nIteration %d " % (i+1))
        wikiqa_train = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path,
                                                 keep_full_embedding=True, text_maxlen=140)

        # dev_list_gen = ListGenerator(dev_file_path, text_maxlen=wikiqa_train.text_maxlen,
        #                              train_word2index=wikiqa_train.word2index,
        #                              additional_word2index=wikiqa_train.additional_word2index,
        #                              oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index,
        #                              train_pad_word_index=wikiqa_train.pad_word_index)

        test_list_gen = ListGenerator(test_file_path, text_maxlen=wikiqa_train.text_maxlen,
                                      train_word2index=wikiqa_train.word2index,
                                      additional_word2index=wikiqa_train.additional_word2index,
                                      oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index,
                                      train_pad_word_index=wikiqa_train.pad_word_index)

        X1_train, X2_train, y_train = wikiqa_train.get_full_batch()
        drmm_tks = DRMM_TKS(embedding=wikiqa_train.embedding_matrix, vocab_size=wikiqa_train.embedding_matrix.shape[0],
                            text_maxlen=wikiqa_train.text_maxlen)

        model = drmm_tks.get_model()
        model.summary()

        optimizer = 'adadelta'

        # either one can be selected. Currently, the choice is manual.
        loss = rank_hinge_loss
        loss = 'mse'

        # validation_data = dev_list_gen.get_list_data()

        # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        # model.fit(x={"query": X1_train, "doc": X1_train}, y=y_train, batch_size=64,
        #           verbose=1, epochs=10, shuffle=True, callbacks=[ValidationCallback(validation_data)])

        data = test_list_gen.get_list_data()
        X1 = data["X1"]
        X2 = data["X2"]
        y = data["y"]
        doc_lengths = data["doc_lengths"]

        predictions = model.predict(x={"query": X1, "doc": X2})

        Y_pred = []
        Y_true = []
        offset = 0

        for doc_size in doc_lengths:
            Y_pred.append(predictions[offset: offset + doc_size])
            Y_true.append(y[offset: offset + doc_size])
            offset += doc_size


        print("MAP: ", mapk(Y_true, Y_pred))
        mets[0] += mapk(Y_true, Y_pred)
        for i, k in enumerate([1, 3, 5, 10, 20]):
            print("nDCG@", str(k), ": ", mean_ndcg(Y_true, Y_pred, k=k))
            mets[i + 1] += mean_ndcg(Y_true, Y_pred, k=k)

        map_val += mapk(Y_true, Y_pred)
        n1 += mean_ndcg(Y_true, Y_pred, k=1)
        n3 += mean_ndcg(Y_true, Y_pred, k=3)
        n5 += mean_ndcg(Y_true, Y_pred, k=5)
        n10 += mean_ndcg(Y_true, Y_pred, k=10)
        n20 += mean_ndcg(Y_true, Y_pred, k=20)

    print([lol/n_iters for lol in mets])

    print("Mean MAP : ", map_val/n_iters)
    print("Mean NDCG@1 : ", n1/n_iters)
    print("Mean NDCG@3 : ", n3/n_iters)
    print("Mean NDCG@5 : ", n5/n_iters)
    print("Mean NDCG@10 : ", n10/n_iters)
    print("Mean NDCG@20 : ", n20/n_iters)