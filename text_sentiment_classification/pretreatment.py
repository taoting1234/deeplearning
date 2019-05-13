import pandas as pd
import numpy as np
import json
import re
from tensorflow import keras


def segmentation(sentence):
    return re.split(r'\W+', sentence)


def sentence_to_vector(data, word_vector, max_len=50):
    vector = np.array([np.array([word_vector.get(j, 0) for j in i.split(' ')]) for i in data])
    return keras.preprocessing.sequence.pad_sequences(vector, maxlen=max_len)


def load_data(filename, test_ratio=0.1):
    df = pd.read_csv(filename, lineterminator='\n')
    data = np.array(df.values)
    train_data_num = int(data.shape[0] * (1 - test_ratio))
    x_train_data = data[0:train_data_num, 1]
    y_train_data = data[0:train_data_num, 2]
    x_test_data = data[train_data_num:, 1]
    y_test_data = data[train_data_num:, 2]

    word_vector = load_word_vector()

    x_train_data = sentence_to_vector(x_train_data, word_vector)
    x_test_data = sentence_to_vector(x_test_data, word_vector)

    y_train_data = np.where(y_train_data == 'Positive', 1, 0)
    y_test_data = np.where(y_test_data == 'Positive', 1, 0)

    return (x_train_data, y_train_data), (x_test_data, y_test_data)


def create_word_vector(filename):
    df = pd.read_csv(filename, lineterminator='\n')
    data = np.array(df.values)
    word_map = dict()
    t = 0
    for i in data[:, 1]:
        for j in segmentation(i):
            if word_map.get(j) is None:
                t += 1
                word_map[j] = t
    return word_map


def save_word_vector(word_map, filename='word_map.json'):
    with open(filename, 'w') as f:
        f.write(json.dumps(word_map))


def load_word_vector(filename='word_map.json'):
    with open(filename) as f:
        return json.loads(f.read())


def get_word_num(filename='word_map.json'):
    return len(load_word_vector(filename))


if __name__ == '__main__':
    save_word_vector(create_word_vector('data/train.csv'))
    # load_data('data/train.csv')
