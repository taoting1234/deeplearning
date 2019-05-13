import tensorflow as tf
from tensorflow import keras

train_epochs = 100
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

model_vgg = keras.applications.VGG16(input_shape=x_train_data.shape[1:], include_top=False)
for layers in model_vgg.layers:
    layers.trainable = False

model = keras.layers.GlobalAveragePooling2D()(model_vgg.output)
# model = keras.layers.Dropout(0.5)(model)
model = keras.layers.Dense(10, activation='softmax')(model)
model = keras.models.Model(inputs=model_vgg.input, outputs=model)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_dataset, epochs=train_epochs, steps_per_epoch=x_train_data.shape[0] // batch_size,
          validation_data=test_dataset, validation_steps=10)
