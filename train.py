import pickle
import tensorflow as tf
import numpy as np
from BLSTM import BLSTM
from util.train_util import get_embedding, get_cross_validation
import argparse
from preprocess import gen_map, get_data_new, write_dict

# python train.py [-p train_path] [-w embedding_path]

print ("Loading the model,please wait a minute...")
gpu_config = "/gpu:0"
num_steps = 100

# 命令行输入参数
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--train_path", help="the path of the train file", default='chatbot.txt')
parser.add_argument("-m", "--model_path", help="the path of the model", default='models/model')
parser.add_argument("-w","--word_emb", help="the word embedding file", default='wiki1.zh.text.vector')
args = parser.parse_args()

train_path = args.train_path
emb_path = args.word_emb
model_path = args.model_path
dict_path = 'index2answer.dict'


X_train, y_train, answer2index = get_data_new(train_path)
index2answer=dict((v,k) for k,v in answer2index.items())
write_dict(index2answer, dict_path)


gen_map(X_train)

word2id = pickle.load(open('word2id', 'rb'))
id2word = pickle.load(open('id2word', 'rb'))

X_train = np.array([[word2id[word] for word in line] for line in X_train])

# 正在读取词向量
embedding_matrix = get_embedding(emb_path, 'word', 400)

print("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.02, 0.02)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BLSTM(400, 20, embedding_matrix, attention=True, num_epochs=100, dropout=0.3)
        print("training model")
        tf.global_variables_initializer().run()
        # 训练的迭代次数
        num = 10
        model.train(sess, model_path, X_train, y_train, X_train, y_train, 0, num_epochs=num)
        print ("success!")