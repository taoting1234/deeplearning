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

train_iterator = train_dataset.make_initializable_iterator()
train_next_element = train_iterator.get_next()
test_iterator = test_dataset.make_initializable_iterator()
test_next_element = test_iterator.get_next()

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')

conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], strides=1, padding='same',
                         activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], strides=1, padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

pred = tf.layers.dense(inputs=dropout, units=10)

loss_function = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_iterator.initializer)
sess.run(test_iterator.initializer)

step = 0
while 1:
    try:
        xs, ys = sess.run(train_next_element)
    except tf.errors.OutOfRangeError:
        break

    sess.run([optimizer], feed_dict={x: xs, y: ys})
    step += 1
    if step % display_step == 0:
        xss, yss = sess.run(test_next_element)
        loss, acc = sess.run([loss_function, accuracy], feed_dict={x: xss, y: yss})
        print('step:{:d} loss:{:.05f} acc:{:.05f}'.format(step, loss, acc))

xss, yss = sess.run(test_next_element)
pred_y = np.argmax(sess.run(pred, {x: xss[:20]}), 1)
true_y = np.argmax(yss[:20], 1)
print('推测的数字', pred_y)
print('真实的数字', true_y)
