#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
import os
import numpy as np
from untils import get_data, create_dictionaries, word2vec_train, loadfile
from model import TextCNN
from sklearn.model_selection import train_test_split

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
x, y = loadfile()
model = word2vec_train(x)
w2indx, w2vec, x = create_dictionaries(model=model, sentences=x)
print('词典大小:',len(w2indx)+1)
embedding_weights = get_data(w2indx, model)
print('样本数量:',x.shape[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=20000)
if os.path.isfile('np_data/x_train.npy') and os.path.isfile('np_data/x_test.npy') and os.path.isfile('np_data/y_train.npy') and os.path.isfile('np_data/y_test.npy'):
    print('load train data and test data')
    x_train = np.load('np_data/x_train.npy')
    x_test = np.load('np_data/x_test.npy')
    y_train = np.load('np_data/y_train.npy')
    y_test = np.load('np_data/y_test.npy')
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=20000)
    np.save('np_data/x_train.npy',x_train)
    np.save('np_data/x_test.npy', x_test)
    np.save('np_data/y_train.npy', y_train)
    np.save('np_data/y_test.npy', y_test)


def main(mode):
    if mode =='train':
        model = TextCNN(embedding_weights)
        model.bulid_graph()
        model.train((x_train,y_train),(x_test,y_test))
    elif mode =='test':
        model = TextCNN(embedding_weights)
        model.bulid_graph()
        model.test((x_test,y_test),'1528038283')


if __name__ == '__main__':
    # main('train')
    main('test')