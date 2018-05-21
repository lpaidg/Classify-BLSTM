import math
from util.my_layer import softmax
import numpy as np
import tensorflow as tf
import pickle
import time
from util.train_util import get_next_batch
from preprocess import simple_process_sentence


class BLSTM(object):
    """
    参考论文中的基于BLSTM-Attention模型来搭建。
    """
    def __init__(self, input_dim, num_steps, embedding_matrix, num_classes=790, is_training=True, num_epochs=10,
                 batch_size=16, hidden_dim=300, learning_rate=0.005, dropout=0, attention=False):
        # 初始化参数
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        if is_training:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.is_training = is_training
        self.word2id = pickle.load(open('word2id', 'rb'))
        # 输入输出
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None])
        # 转化输入
        self.one_hot = tf.one_hot(self.targets, self.num_classes, dtype=tf.float32)
        self.embedding_matrix = tf.Variable(embedding_matrix, dtype=tf.float32)
        self.input_emb = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)
        self.input_emb = tf.cast(self.input_emb, tf.float32)

        # 计算输入长度
        self.mask = tf.sign(self.inputs)
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # embedding dropout
        if self.is_training:
            self.input_emb = tf.nn.dropout(self.input_emb, keep_prob=1 - self.dropout)

        self.input_emb = tf.transpose(self.input_emb, [1, 0, 2])
        self.input_emb = tf.reshape(self.input_emb, [-1, self.input_dim])
        self.input_emb = tf.split(self.input_emb, self.num_steps, 0)
        # if self.is_training:
        #     self.input_emb = tf.nn.dropout(self.input_emb, keep_prob=1 - self.dropout)

        self.input_shape = tf.shape(self.input_emb)
        # 双向LSTM单元
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True, use_peepholes=True)
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True, use_peepholes=True)
        # 如果是在训练就dropout
        if self.is_training:
            lstm_cell_fw =\
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout))

        # LSTM层
        self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.input_emb,
            dtype=tf.float32,
            sequence_length=self.length
        )

        self.outputs = tf.transpose(self.outputs, [1, 0, 2])

        if not attention:
            self.fw, self.bw = tf.split(self.outputs, 2, 2)
            self.added = tf.add(self.fw, self.bw)
            self.added = tf.reshape(self.added, [self.batch_size, self.num_steps, 1, self.hidden_dim])
            self.maxed = tf.nn.max_pool(self.added, [1, self.num_steps, 1, 1], [1, 1, 1, 1], padding='VALID')
            self.maxed = tf.reshape(self.maxed, [self.batch_size, self.hidden_dim])
            self.softmax_w = tf.get_variable('softmax_w', dtype=tf.float32, shape=[self.hidden_dim, self.num_classes])
            self.softmax_b = tf.get_variable('softmax_b', dtype=tf.float32, shape=[self.num_classes])
            self.softmax_output = tf.nn.softmax(tf.matmul(self.maxed, self.softmax_w) + self.softmax_b)
            self.loss = tf.reduce_mean(tf.reduce_mean(- self.one_hot * tf.log(self.softmax_output), reduction_indices=[1]))
            self.predict = tf.arg_max(self.softmax_output, dimension=1)
            self.optimizer_list = [tf.train.AdamOptimizer(0.001).minimize(self.loss),
                                   tf.train.RMSPropOptimizer(0.001).minimize(self.loss),
                                   tf.train.MomentumOptimizer(0.001, 0.9).minimize(self.loss),
                                   tf.train.MomentumOptimizer(0.00000001, 0.9).minimize(self.loss)]
            self.optimizer = self.optimizer_list[0]
        else:
            # if self.is_training:
            #     self.outputs = tf.nn.dropout(self.outputs, keep_prob=0.5)
            self.M = tf.tanh(self.outputs)
            self.attention_w = tf.get_variable('attention_w', [self.hidden_dim * 2], dtype=tf.float32)
            self.attention_w_batch = tf.reshape(tf.concat( [self.attention_w] * self.batch_size,0), [self.batch_size, self.hidden_dim *2, 1])
            self.alpha = tf.matmul(self.M, self.attention_w_batch)
            self.alpha = tf.reshape(self.alpha, [self.batch_size, self.num_steps])
            self.alpha = softmax(self.alpha)
            self.outputs = tf.transpose(self.outputs, [0, 2, 1])
            self.alpha = tf.reshape(self.alpha, [self.batch_size, self.num_steps, 1])
            self.r = tf.matmul(self.outputs, self.alpha)
            self.r = tf.reshape(self.r, [self.batch_size, self.hidden_dim * 2])
            self.h = tf.tanh(self.r)
            self.softmax_w = tf.get_variable('softmax_w', dtype=tf.float32, shape=[self.hidden_dim * 2, self.num_classes])
            self.softmax_b = tf.get_variable('softmax_b', dtype=tf.float32, shape=[self.num_classes])
            self.softmax_output = tf.nn.softmax(tf.matmul(self.h, self.softmax_w) + self.softmax_b)
            self.loss = tf.reduce_mean(tf.reduce_mean(- self.one_hot * tf.log(self.softmax_output), reduction_indices=[1])) +\
                        1e-7 * tf.nn.l2_loss(self.attention_w) + \
                        1e-7 * tf.nn.l2_loss(self.softmax_w) + \
                        1e-7 * tf.nn.l2_loss(self.softmax_b)
                        # 1e-5 * tf.nn.l2_loss(self.dp_trans) +\
                        # 1e-5 * tf.nn.l2_loss(self.dp_trans_b) + 1e-5 * tf.nn.l2_loss(self.rel_map)

            self.predict = tf.arg_max(self.softmax_output, dimension=1)
            self.optimizer_list = [tf.train.AdamOptimizer(0.001).minimize(self.loss),
                                   tf.train.RMSPropOptimizer(0.001).minimize(self.loss),
                                   tf.train.MomentumOptimizer(0.001, 0.9).minimize(self.loss),
                                   tf.train.MomentumOptimizer(0.00000001, 0.9).minimize(self.loss)]
            self.optimizer = self.optimizer_list[0]
            # self.sgd = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)

    def train(self, sess, save_file, X_train, y_train, X_val, y_val, cross_validation, num_epochs=50):
        # 用来得到id到词的映射和id到标签的映射。
        self.batch_size = 16
        max_f1 = 0
        saver = tf.train.Saver()
        self.num_epochs = num_epochs
        # 获得迭代次数，向上取整。
        loss_list = [10 for i in range(10)]
        self.optimizer = self.optimizer_list[0]
        for epoch in range(self.num_epochs):
            num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
            # shuffle train in each epoch
            # 打乱训练集
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            # 开始训练。
            self.is_training = True
            train_targets = []
            train_predicts = []
            loss_sum = 0
            print("current epoch: %d, cross_validation: %d" % (epoch, cross_validation))
            start_time = time.time()
            for iteration in range(num_iterations):
                # train
                # 获得下一批数据
                self.is_training = True
                X_train_batch, y_train_batch = get_next_batch([X_train, y_train], start_index=iteration * self.batch_size, batch_size=self.batch_size)
                predict, loss, _ =\
                    sess.run([
                        # self.input_shape
                        self.predict,
                        self.loss,
                        self.optimizer
                    ],
                        feed_dict={
                        self.inputs: X_train_batch,
                        self.targets: y_train_batch
                    })
                train_targets += list(y_train_batch)
                train_predicts += (list(predict))
                loss_sum += loss

                # 计算测试集F1
                #if iteration % 10 == 0:###########
                    #print('已计算%d个数据' % (iteration * self.batch_size))###########
                    # self.is_training = False
                    # val_num_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))
                    # targets = []
                    # predicts = []
                    # for val_iteration in range(val_num_iterations):
                    #     # train
                    #     # 获得下一批数据
                    #     X_val_batch, y_val_batch = get_next_batch([X_val, y_val], start_index=val_iteration * self.batch_size, batch_size=self.batch_size)
                    #
                    #     predict = sess.run(
                    #                 # self.input_shape
                    #                 self.predict
                    #                 ,
                    #                 feed_dict={
                    #                     self.inputs: X_val_batch,
                    #                     self.targets: y_val_batch
                    #                 }
                    #     )
                    #     targets += list(y_val_batch)
                    #     predicts += list(predict)
                    # acc = sum([1 if targets[idx] == predicts[idx] else 0 for idx in range(len(targets))])/len(targets)
                    # if acc > max_f1:
                    #     max_f1 = acc
                    # print('验证集准确率为%f' % acc)
            loss_sum /= num_iterations

            print('loss:' + str(loss_sum))
            end_time = time.time()
            print('用时%fs' % (end_time - start_time))
            if epoch % 1 == 0:
                print('计算训练集准确率')
                acc = sum([1 if train_targets[idx] == train_predicts[idx] else 0 for idx in range(len(train_targets))])/len(train_targets)
                print('训练集准确率为%f' % acc)

                # 计算测试集F1
                # print('计算验证集F1')
                # self.is_training = False
                # val_num_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))
                # targets = []
                # predicts = []
                # for val_iteration in range(val_num_iterations):
                #     # train
                #     # 获得下一批数据
                #     X_val_batch, y_val_batch = get_next_batch([X_val, y_val], start_index=val_iteration * self.batch_size, batch_size=self.batch_size)
                #
                #     predict = sess.run(
                #                 # self.input_shape
                #                 self.predict
                #                 ,
                #                 feed_dict={
                #                     self.inputs: X_val_batch,
                #                     self.targets: y_val_batch
                #                 }
                #     )
                #     targets += list(y_val_batch)
                #     predicts += list(predict)
                # acc = sum([1 if targets[idx] == predicts[idx] else 0 for idx in range(len(targets))])/len(targets)
                # if acc > max_f1:
                #     max_f1 = acc
                # print('验证集准确率为%f' % acc)
        print('保存模型中..')
        saver.save(sess, save_file)
        # print('the max f1 is ' + str(max_f1))
        # with open('log.txt', 'a') as log_output:
        #     log_output.write('the max f1 is ' + str(max_f1) + '\n')

    def predict_label(self, sess, string_input):
        self.is_training = False
        if len(string_input) > 100:
            return 1
        else:
            model_input = simple_process_sentence(string_input)
            model_input = np.array([[self.word2id[i] if i in self.word2id.keys() else self.word2id['<new>'] for i in model_input]])
            pred, softmax_output = sess.run(
                [self.predict,self.softmax_output],
                feed_dict={
                    self.inputs: model_input,
                    self.targets: np.array([0])
                }
            )
            return pred[0], softmax_output


