import tensorflow as tf
from tensorflow import keras
from text_sentiment_classification.pretreatment import load_data

train_epochs = 10
batch_size = 10
test_batch_size = 100
learning_rate = 0.001

test_ratio = 0.1
min_count = 5
vector_size = 100
max_len = 100
input_length = 100

(x_train_data, y_train_data), (x_test_data, y_test_data), n_symbols, embedding_weights = \
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

model = keras.Sequential()  # or Graph or whatever
model.add(keras.layers.Embedding(output_dim=vector_size,
                                 input_dim=n_symbols,
                                 mask_zero=True,
                                 weights=[embedding_weights],
                                 input_length=input_length))  # Adding Input Length
model.add(keras.layers.LSTM(128, activation='tanh'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))  # Dense=>全连接层,输出维度=3
model.add(keras.layers.Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
          validation_data=test_dataset, validation_steps=10)

pass
