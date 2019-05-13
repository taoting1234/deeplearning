import pandas as pd
import numpy as np
import json
import re

from gensim.models import word2vec
from tensorflow import keras


def segmentation(sentence):
    return re.split(r'\W+', sentence)


def sentence_to_vector(data, word_vector, max_len=50):
    vector = []
    for i in data:
        tmp = []
        for j in segmentation(i):
            try:
                tmp.append(word_vector.wv.get_vector(j))
            except KeyError:
                tmp.append(np.zeros(100))
        tmp = np.array(tmp)
        vector.append(tmp)
    vector = np.array(vector)
    return keras.preprocessing.sequence.pad_sequences(vector, maxlen=max_len, dtype='float64')


def load_data(filename, test_ratio=0.1):
    df = pd.read_csv(filename, lineterminator='\n')
    data = np.array(df.values)
    train_data_num = int(data.shape[0] * (1 - test_ratio))
    x_train_data = data[0:train_data_num, 1]
    y_train_data = data[0:train_data_num, 2]
    x_test_data = data[train_data_num:, 1]
    y_test_data = data[train_data_num:, 2]

    word_vector = create_word2vec_model(filename)

    x_train_data = sentence_to_vector(x_train_data, word_vector)
    x_test_data = sentence_to_vector(x_test_data, word_vector)

    y_train_data = np.where(y_train_data == 'Positive', 1, 0)
    y_test_data = np.where(y_test_data == 'Positive', 1, 0)

    return (x_train_data, y_train_data), (x_test_data, y_test_data)


def create_word2vec_model(filename):
    df = pd.read_csv(filename, lineterminator='\n')
    data = np.array(df.values)
    sentences = [segmentation(i) for i in data[:, 1]]
    model = word2vec.Word2Vec(sentences, min_count=5)
    return model


def save_word2vec_model(model, filename='word2vec.model'):
    model.save(filename)


def load_word2vec_model(filename):
    return word2vec.Word2Vec.load(filename)


if __name__ == '__main__':
    load_data('data/train.csv')
