import tensorflow as tf
from tensorflow import keras
from text_sentiment_classification.pretreatment import load_data, get_word_num

train_epochs = 10
batch_size = 100
test_batch_size = 100
learning_rate = 0.001
display_step = 10

(x_train_data, y_train_data), (x_test_data, y_test_data) = load_data('data/train.csv')

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
model.add(keras.layers.Embedding(get_word_num() + 1, 256))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='relu'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
          validation_data=test_dataset, validation_steps=10)

pass
