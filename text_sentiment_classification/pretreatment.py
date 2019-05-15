import pandas as pd
import numpy as np
import re

from gensim.corpora import Dictionary
from gensim.models import word2vec
from tensorflow import keras


def segmentation(sentence):
    res = re.split(r'[\W\d]+', sentence)
    for i, j in enumerate(res):
        if j == '':
            res.pop(i)
            break
    return res


def load_data(filename, test_ratio=0.1, min_count=5, vector_size=100, max_len=50):
    df = pd.read_csv(filename, lineterminator='\n')
    data = np.array(df.values)
    train_data_num = int(data.shape[0] * (1 - test_ratio))
    x_train_data = data[0:train_data_num, 1]
    y_train_data = data[0:train_data_num, 2]

    x_test_data = data[train_data_num:, 1]
    y_test_data = data[train_data_num:, 2]

    model = create_word2vec_model([segmentation(i) for i in x_train_data], min_count, vector_size)
    index_dict, word_vectors, x_train_data = create_dictionaries(model, [segmentation(i) for i in x_train_data],
                                                                 max_len)
    _, _, x_test_data = create_dictionaries(model, [segmentation(i) for i in x_test_data], max_len)

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vector_size))  # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    y_train_data = np.where(y_train_data == 'Positive', 1, 0)
    y_test_data = np.where(y_test_data == 'Positive', 1, 0)

    return (x_train_data, y_train_data), (x_test_data, y_test_data), n_symbols, embedding_weights


def create_dictionaries(model, combined, max_len=50):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    #  freqxiao10->0 所以k+1
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
    w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

    data = []
    for sentence in combined:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)  # freqxiao10->0
        data.append(new_txt)

    combined = keras.preprocessing.sequence.pad_sequences(data, maxlen=max_len)
    return w2indx, w2vec, combined


def create_word2vec_model(combined, min_count=5, vector_size=100):
    model = word2vec.Word2Vec(combined, min_count=min_count, size=vector_size)
    return model


if __name__ == '__main__':
    load_data('data/train.csv')
    pass
