#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
import logging

vocab_dim = 100
keep_prob= 0.75
maxlen = 150
n_iterations = 1  # ideally more..
n_exposures = 3
test_size=0.1
window_size = 7
num_filters= 32
batch_size = 64
layer_num = 2
LR = 0.001
train_rate = 0.9
n_epoch =60
cpu_count = multiprocessing.cpu_count()

#read xlsx
def read_xlsx(path):
    return pd.read_excel(path, encoding='gbk').loc[:, ['tokenizer', 'sentiment']]


# 在get_x中把str转化成list
def process(x):
    return np.array(list((map(eval, x))))


def get_x(df):
    # squeeze 去除维度为1的维。
    return np.squeeze(df.iloc[:, 0:1].apply(process, axis=0).values)


def transe_one_hot(x):
    x = x.iloc[:, 1:].values.flatten()
    one_hot = []
    for i in x:
        if i == 2:
            one_hot.append([1, 0])
        else:
            one_hot.append([0, 1])
    return np.array(one_hot, dtype=np.float32)


# 加载训练文件
def loadfile():
    if os.path.isfile('np_data/x.npy') and os.path.isfile('np_data/y.npy'):
        x = np.load('np_data/x.npy')
        y = np.load('np_data/y.npy')
    else:
        df_label = read_xlsx('./wdzj_baidu_senti/label.xlsx')
        x = get_x(df_label)
        y = transe_one_hot(df_label)
        np.save('np_data/x.npy', x)
        np.save('np_data/y.npy', y)
    return x, y

#把句子变成索引
def parse_dataset(sentences, w2indx):
    data = []
    for sentence in sentences:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    # 这句话要特别注意，意思是现在得到句子关于词的index向量，下面这个函数maxlen意思是句子最大长度位100个词，
    # 例如句子index向量为[107  644  369    7  412 ...],补成 [ 107 644 369 7 412 ...0 0 0 0 ]
    # padding=post 句子长度不够在句尾补零， truncating=post 句子过长截断尾部。
    data = sequence.pad_sequences(data, maxlen=maxlen, padding='post', truncating='post')
    return data

# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    if os.path.isfile('word2vec_data/Word2vec_model.pkl'):
        print('load the w2v_model')
        model = Word2Vec.load('word2vec_data/Word2vec_model.pkl')
    else:
        model = Word2Vec(combined, size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         iter=n_iterations)
        print('saving the w2v.model')
        model.save('word2vec_data/Word2vec_model.pkl')
    return model

def create_dictionaries(model=None, sentences=None):

    if (sentences is not None) and (model is not None):
        #这个包可以给文档中的词赋予id
        gensim_dict = Dictionary()
        #model.wv.vocab dict key:次 value:gensim.models.keyedvectors.Vocab object
        #所以keys()获得所有的词
        #doc2bow可以看https://www.douban.com/note/620615113/，allow_update会给字典中没出现过的赋予新的id(因为Dictionary中可以初始化文档)
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        #k 词，v id
        w2indx = {k: v + 1 for k, v in gensim_dict.token2id.items()}  # 所有频数超过10的词语的索引
        #model[word]词向量
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量
        #sentences是句子索引
        sentences = parse_dataset(sentences,w2indx)

        return w2indx, w2vec, sentences
    else:
        print('No data provided...')

def get_data(w2indx, model):
    if os.path.isfile('np_data/embedding_weights.npy'):
        embedding_weights = np.load('np_data/embedding_weights.npy')
    else:
        n_symbols = len(w2indx) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
        for word, index in w2indx.items():#从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = model[word]
        np.save('np_data/embedding_weights.npy', embedding_weights)
    return embedding_weights

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
