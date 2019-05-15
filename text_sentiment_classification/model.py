import tensorflow as tf
from tensorflow import keras
from text_sentiment_classification.pretreatment import load_data

train_epochs = 10
batch_size = 100
test_batch_size = 100
learning_rate = 0.001

network_type = 4

test_ratio = 0.1  # 测试集的比例
min_count = 5  # 去掉出现次数小于min_count的词
vector_size = 10  # 词向量的大小
max_len = 30  # 一个句子中最多的词

(x_train_data, y_train_data), (x_test_data, y_test_data), vocab_size, embedding_weights = \
    load_data('data/train.csv', test_ratio, min_count, vector_size, max_len)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data, y_train_data)) \
    .map(lambda a, b: (a, tf.one_hot(b, 2))) \
    .shuffle(1000).repeat(train_epochs).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data, y_test_data)) \
    .map(lambda a, b: (a, tf.one_hot(b, 2))) \
    .shuffle(1000).repeat().batch(test_batch_size)

train_iterator = train_dataset.make_initializable_iterator()
train_next_element = train_iterator.get_next()
test_iterator = test_dataset.make_initializable_iterator()
test_next_element = test_iterator.get_next()

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=vocab_size,  # 词个数
                                 output_dim=vector_size,  # 词向量大小
                                 # mask_zero=True,
                                 weights=[embedding_weights],  # 权重
                                 input_length=max_len))  # 句子最大包含多少个词
if network_type == 1:  # cnn
    model.add(keras.layers.Convolution1D(256, 3, padding="same"))
    model.add(keras.layers.MaxPool1D(3, 3, padding="same"))
    model.add(keras.layers.Convolution1D(128, 3, padding="same"))
    model.add(keras.layers.MaxPool1D(3, 3, padding="same"))
    model.add(keras.layers.Convolution1D(64, 3, padding="same"))
    model.add(keras.layers.MaxPool1D(3, 3, padding="same"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation="relu"))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
elif network_type == 2:  # rnn
    model.add(keras.layers.GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
elif network_type == 3:  # 双向rnn
    model.add(keras.layers.Bidirectional(keras.layers.GRU(
        256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.GRU(
        256, dropout=0.2, recurrent_dropout=0.1)))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
elif network_type == 4:  # cnn+rnn
    model.add(keras.layers.Convolution1D(256, 3, padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool1D(pool_size=2))
    model.add(keras.layers.GRU(256, dropout=0.2, recurrent_dropout=0.1))
    model.add(keras.layers.Dense(2, activation="sigmoid"))
elif network_type == 5:  # cnn+双向rnn
    model.add(keras.layers.Convolution1D(256, 3, padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool1D(pool_size=2))
    model.add(keras.layers.Bidirectional(keras.layers.GRU(
        256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.GRU(
        256, dropout=0.2, recurrent_dropout=0.1)))
    model.add(keras.layers.Dense(2, activation="sigmoid"))

model.summary()

# model = keras.utils.multi_gpu_model(model, gpus=4)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
          validation_data=test_dataset, validation_steps=10)

pass
