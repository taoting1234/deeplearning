import tensorflow as tf
import numpy as np

# 设置超参数
train_epochs = 1000
batch_size = 10
learning_rate = 0.01
display_step = 100
# 生成数据
x_data = np.linspace(0, 100, 500)
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 20
# 生成dataset
x_placeholder = tf.placeholder(tf.float32, x_data.shape)
y_placeholder = tf.placeholder(tf.float32, y_data.shape)
dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder)) \
    .shuffle(100).repeat(train_epochs).batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
# 定义网络结构
x, y = next_element
w = tf.Variable(1.0, name='w')
b = tf.Variable(0.0, name='b')
pred = w * x + b
# 定义损失函数和优化器
loss_function = tf.losses.mean_squared_error(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
# 初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(iterator.initializer, feed_dict={x_placeholder: x_data, y_placeholder: y_data})
# 训练
step = 0
loss_sum = 0
while 1:
    try:
        _, loss = sess.run([optimizer, loss_function])
    except tf.errors.OutOfRangeError:
        break

    step += 1
    loss_sum += loss
    if step % display_step == 0:
        print('step:{:03d} loss:{:.10f} w:{:.5f} b:{:.5f}'.format(step, loss_sum / display_step, sess.run(w),
                                                                  sess.run(b)))
        loss_sum = 0

sess.close()
