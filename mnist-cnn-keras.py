import tensorflow as tf
import numpy as np
from tensorflow import keras

train_epochs = 1
batch_size = 100
test_batch_size = 100
learning_rate = 0.01
display_step = 10

(x_train_data, y_train_data), (x_test_data, y_test_data) = keras.datasets.mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data, y_train_data)) \
    .map(lambda a, b: (tf.reshape(a, (28, 28, 1)) / 255, tf.one_hot(b, 10))) \
    .shuffle(1000).repeat(train_epochs).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data, y_test_data)) \
    .map(lambda a, b: (tf.reshape(a, (28, 28, 1)) / 255, tf.one_hot(b, 10))) \
    .shuffle(1000).repeat().batch(test_batch_size)

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(rate=0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

parallel_model = keras.utils.multi_gpu_model(model, gpus=4)

parallel_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])
parallel_model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
                   validation_data=test_dataset, validation_steps=10)
