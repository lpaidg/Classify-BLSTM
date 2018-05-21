import pickle
import re
import numpy as np
import jieba

def load_data(train_path):
    X_train, y_train, answer2index = get_data_new(train_path)
    gen_map(X_train)
    word2id = pickle.load(open('word2id', 'rb'))
    id2word = pickle.load(open('id2word', 'rb'))

    X_train = np.array([[word2id[word] for word in line] for line in X_train])

def build_map(tokens, name):
    """
    生成id和tokens的映射
    :param tokens: 输入的一个序列，为一个列表，不带重复数据
    :param name: tokens的名称，用来保存为相应的文件
    :return:
    """
    id2tokens = {0: '<pad>'}
    tokens2id = {'<pad>': 0}
    count = 1
    for token in tokens:
        id2tokens[count] = token
        tokens2id[token] = count
        count += 1
    id2tokens[count] = '<new>'
    tokens2id['<new>'] = count
    pickle.dump(id2tokens, open('id2' + name, 'wb'))
    pickle.dump(tokens2id, open(name + '2id', 'wb'))

def gen_map(sentences, simple=True):
    """
    生成map
    :param sentences: 输入的句子
    :return:
    """
    if simple:
        all_word = []
        for i in sentences:
            for j in i:
                if j not in all_word and j != '<pad>':
                    all_word.append(j)
        build_map(all_word, 'word')
    else:
        all_word = []
        for i in sentences:
            for j in i:
                for k in j:
                    if k not in all_word and k != '<pad>':
                        all_word.append(k)
        build_map(all_word, 'word')

def get_data_new(path):
    """
    得到数据
    :param path:数据路径
    :return:
    """
    lines = open(path).read().strip().split('\n')
    pairs = [[s for s in l.split('    ')] for l in lines]
    splited_sentences = []
    answer_index = []
    answer2index = {}
    index=1
    for i in range(len(pairs)):
        segs = simple_process_sentence(pairs[i][0])
        splited_sentences.append(segs)
        if pairs[i][1] not in answer2index:
            answer2index[pairs[i][1]] = index
            index = index+1
        answer_index.append(answer2index[pairs[i][1]])

    return [np.array(splited_sentences), np.array(answer_index),answer2index]

def segment(string):
    seg = jieba.cut(string)
    seg_list = []
    for i in seg:
        if i != '' and i != ' ' and i != '\n' and i != '?':
            seg_list.append(i)
    return seg_list


def split_sentence(string):
    pieces = re.split('，|。|？|！|,', string)
    return pieces


def padding(arr):
    while len(arr) < 20:############
        arr.append('<pad>')
    return arr

def simple_process_sentence(string):
    r = '[？，。]'
    string = re.sub(r, '', string)
    line = string
    seg = segment(line)
    seg = padding(seg)
    return seg

def load_data(train_path):
    X_train, y_train, answer2index = get_data_new(train_path)
    gen_map(X_train)
    word2id = pickle.load(open('word2id', 'rb'))
    id2word = pickle.load(open('id2word', 'rb'))

    X_train = np.array([[word2id[word] for word in line] for line in X_train])