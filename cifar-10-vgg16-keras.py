import tensorflow as tf
from tensorflow import keras

train_epochs = 10
batch_size = 100
test_batch_size = 100
learning_rate = 0.01
display_step = 10

(x_train_data, y_train_data), (x_test_data, y_test_data) = keras.datasets.cifar10.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_data, y_train_data)) \
    .map(lambda a, b: (a / 255, tf.reshape(tf.one_hot(b, 10), [-1]))) \
    .shuffle(1000).repeat(train_epochs).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_data, y_test_data)) \
    .map(lambda a, b: (a / 255, tf.reshape(tf.one_hot(b, 10), [-1]))) \
    .shuffle(1000).repeat().batch(test_batch_size)

model = tf.keras.Sequential()

model.add(keras.applications.VGG16(input_shape=x_train_data.shape[1:], include_top=False))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation='softmax'))

model.layers[0].trainable = False  # vgg不训练

model.summary()

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
          validation_data=test_dataset, validation_steps=10)
