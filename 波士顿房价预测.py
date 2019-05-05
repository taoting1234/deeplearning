import tensorflow as tf
import numpy as np
import pandas as pd

train_epochs = 1000
batch_size = 10
learning_rate = 0.01
display_step = 100

df = pd.read_csv('data/boston.csv')
df.describe()

data = np.array(df.values)
x_data = data[:, :12]
y_data = data[:, 12].reshape([-1, 1])
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)) \
    .shuffle(1000).repeat(train_epochs).batch(batch_size)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

x = tf.placeholder(tf.float32, [None, 12], name='x')
y = tf.placeholder(tf.float32, [None, 1], name='y')

w = tf.Variable(tf.truncated_normal([12, 1], stddev=0.01), name='w')
b = tf.Variable(0.0, name='b')
pred = tf.matmul(x, w) + b

loss_function = tf.reduce_mean(tf.pow(y - pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer)

step = 0
while 1:
    try:
        xs, ys = sess.run(next_element)
    except tf.errors.OutOfRangeError:
        break

    sess.run([optimizer], feed_dict={x: xs, y: ys})
    step += 1
    if step % display_step == 0:
        loss = sess.run(loss_function, feed_dict={x: xs, y: ys})
        print('step:{:d} loss:{:.05f}'.format(step, loss))
