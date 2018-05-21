#-*- coding:utf-8 -*-
import numpy as np
import jieba
import re
from util.preprocess_util import build_map

def write_dict(index2answer, dict_path):
    f = open(dict_path, 'a')
    for a in index2answer:
        f.write(str(a) + "\t" + (index2answer[a]) )
        f.write("\n")
    f.close()

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


def process_sentence(string):
    line = string
    splited = split_sentence(line.rstrip().lower())
    segs = []
    for j in splited:
        if j != '':
            seg = segment(j)
            seg = padding(seg)
            segs.append(seg)
    while len(segs) < 1:
        segs.append(['<pad>' for _ in range(10)])
    return segs


def simple_process_sentence(string):
    r = '[？，。]'
    string = re.sub(r, '', string)
    line = string
    seg = segment(line)
    seg = padding(seg)
    return seg


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
        if 20 > len(pairs[i][0]) > 0:
            segs = simple_process_sentence(pairs[i][0])
            splited_sentences.append(segs)
            if pairs[i][1] not in answer2index:
                answer2index[pairs[i][1]] = index
                index = index+1
            answer_index.append(answer2index[pairs[i][1]])
    return [np.array(splited_sentences), np.array(answer_index),answer2index]


def get_dict(path):
    lines = open(path).read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    index2answer = {}
    for i in range(len(pairs)):
        index2answer[int(pairs[i][0])] = pairs[i][1]
    return index2answer

if __name__ == '__main__':
    get_data_new()
    # gen_map()
