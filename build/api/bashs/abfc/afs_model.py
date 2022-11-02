import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.tools import freeze_graph


def build(total_batch, input_size, output_size):
    X = tf.placeholder(tf.float32, [None, input_size])  # [512,]
    Y = tf.placeholder(tf.float32, [None, output_size])  # [5,]
    global_step = tf.Variable(0, trainable=False)  # 创建变量，训练时不更新值

    tf.add_to_collection('input', X)  # 添加X到列表input
    tf.add_to_collection('output', Y)

    with tf.variable_scope('attention_module') as scope:  # 指定变量作用域
        E_W = tf.Variable(tf.truncated_normal([input_size, 32], stddev=0.1, seed=1))  # 正态分布相同变量[512,32]
        E_b = tf.Variable(tf.constant(0.1, shape=[32]))  # 创建0.1填充的长度为隐藏单元数目的变量,[32,]

        E = tf.nn.tanh(tf.matmul(X, E_W) + E_b)  # 使用密集网络提取原始输入之间的内在关系（1）式,[32,]

        A_W = tf.Variable(tf.truncated_normal([input_size, 32, 2], stddev=0.1, seed=1))  # [512,32,2]
        A_b = tf.Variable(tf.constant(0.1, shape=[input_size, 2]))  # [512,2]

        A_W_unstack = tf.unstack(A_W, axis=0)  # 将A_W沿着0维分解,512个[32,2]
        A_b_unstack = tf.unstack(A_b, axis=0)  # 512个[2,]

        # 每个特征选择/未选择的概率
        attention_out_list = []
        for i in range(input_size):
            attention_FC = tf.matmul(E, A_W_unstack[i]) + A_b_unstack[i]  # [2,]，（2）、（3）式
            attention_out = tf.nn.softmax(attention_FC)  # 特征选择/未选择的概率，[2,]，（4）式
            attention_out = tf.expand_dims(attention_out[:, 1], axis=1)  # 被选择的概率，[1,]
            attention_out_list.append(attention_out)
        A = tf.squeeze(tf.stack(attention_out_list, axis=1), axis=2)  # [512,]

    with tf.variable_scope("learning_module") as scope:
        G = tf.multiply(X, A)  # 加权特征（5）式,[512,]
        L_W1 = tf.Variable(
            tf.truncated_normal([input_size, 500], stddev=0.1, seed=1))  # 正态分布相同变量[512,500]
        L_b1 = tf.Variable(tf.constant(0.1, shape=[500]))  # [500,]
        L_W2 = tf.Variable(
            tf.truncated_normal([500, output_size], stddev=0.1, seed=1))  # [500,5]
        L_b2 = tf.Variable(tf.constant(0.1, shape=[output_size]))  # [5,]

        variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 滑动平均
        # L_FC = tf.nn.leaky_relu(tf.layers.dense(G, 500) + L_b1)
        L_FC = tf.nn.tanh(tf.matmul(G, L_W1) + L_b1)  # [500,]
        # O = tf.layers.dense(L_FC, 5) + L_b2  # [5,]
        O = tf.matmul(L_FC, L_W2) + L_b2  # [5,]

        average_L_FC = tf.nn.relu(
            tf.matmul(G, variable_averages.average(L_W1)) + variable_averages.average(L_b1))  # [500,]
        average_O = tf.matmul(average_L_FC, variable_averages.average(L_W2)) + variable_averages.average(L_b2)  # [5,]

    with tf.name_scope("Loss") as scope:
        # regularizer = tf.contrib.layers.l2_regularizer(FLAGS.regularization_rate)  # 正则化，平方的和/行数
        regularizer = tf.keras.regularizers.L2(0.0001)  # 正则化，平方的和/行数
        regularization = regularizer(L_W1) + regularizer(L_W2)

        learning_rate = tf.train.exponential_decay(
            0.8, global_step, total_batch,
            0.99)  # 学习率应用指数衰减

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=O, labels=tf.argmax(Y, 1))  # 求稀疏交叉熵
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + regularization

        correct_prediction = tf.equal(tf.argmax(average_O, 1), tf.argmax(Y, 1))  # 逐个元素判断是否相等
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 转换数据类型后求均值
    with tf.name_scope("Train") as scope:
        vars_A = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='attention_module')
        vars_L = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='learning_module')
        vars_R = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Loss')
        # Minimizing Loss Function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step, var_list=[vars_A, vars_L, vars_R])

    with tf.control_dependencies([optimizer, variable_averages_op]):  # 当括号中的参数执行完毕再执行with
        train_op = tf.no_op(name='train')
    for op in [train_op, A]:
        tf.add_to_collection('train_ops', op)
    for op in [loss, accuracy]:
        tf.add_to_collection('validate_ops', op)