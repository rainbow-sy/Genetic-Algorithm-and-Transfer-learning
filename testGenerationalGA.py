# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 15:54 
@file: testGenerationalGA.py
@author: Matt
"""
from DataMgr import load_cifar10, write_performance
from GenerationalGA import GenerationalGA
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
#from DataNew import load_data

#root = 'D:/datasets/cifar10'
root = r'E:\遗传算法_迁移学习 -VGG\数据集\cifar10'
# root = '/home/u800199/workdir/datasets/cifar10'
X_train, y_train, X_test, y_test = load_cifar10(root)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# X, Y = load_data(root)
# X_train, X_test = X[0:80], X[80:100]
# y_train, y_test = Y[0:80], Y[80:100]
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

train_size = len(X_train)
test_size = len(X_test)
g = GenerationalGA(
    _X_train=X_train[:train_size],
    _y_train=y_train[:train_size],
    _X_test=X_test[:test_size],
    _y_test=y_test[:test_size],
    _pop_size=20,
    _r_mutation=0.1,
    _p_crossover=0,  # no use
    _p_mutation=0.001,
    _max_iter=10,
    _min_fitness=0.95,
    _batch_size=5000,
    _elite_num=0,  # no use
    _mating_pool_size=0,  # no use
)
g.run()

write_performance(g.evaluation_history, 'GenerationalGA_CIFAR10.txt')
