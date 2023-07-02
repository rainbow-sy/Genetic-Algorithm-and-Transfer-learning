# -*- coding: utf-8 -*-
"""
Created on 2019/11/20 23:13 
@file: GA.py
@author: Matt
"""
from abc import abstractmethod

import numpy as np
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
#from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras import Model
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
#from keras.models import Model
#from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras

class GA:
    def __init__(self, _X_train, _y_train, _X_test, _y_test, _pop_size, _r_mutation, _p_crossover, _p_mutation,
                 _max_iter, _min_fitness, _elite_num, _mating_pool_size, _batch_size=32, _dataset='cifar10'):
        # input params
        self.X_train = _X_train
        self.y_train = _y_train
        self.X_test = _X_test
        self.y_test = _y_test
        self.pop_size = _pop_size
        self.r_mutation = _r_mutation
        self.p_crossover = _p_crossover  # for steady-state
        self.p_mutation = _p_mutation  # for generational
        self.max_iter = _max_iter
        self.min_fitness = _min_fitness
        self.elite_num = _elite_num  # for elitism
        self.mating_pool_size = _mating_pool_size  # for elitism
        self.batch_size = _batch_size
        self.dataset = _dataset
        # other params
        self.chroms = []#表现型，即不同的model
        self.evaluation_history = []
        self.stddev = 0.5
        self.loss_func = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        #self.gene = np.random.randint(2, size=(self.pop_size, 94))
        #gene里边装的是gene型，chroms里边装的是表现型（model）
        self.gene = self.gene_create(gene_num = self.pop_size)
    @property
    def cur_iter(self):
        return len(self.evaluation_history)

    def shuffle_batch(self):
        series = list(range(len(self.X_train)))
        np.random.shuffle(series)     #打乱顺序函数
        return series

    def gene_create(self, gene_num = 20):
        gene = np.zeros((gene_num, 13))
        # randnum=np.random.randint(94)#生成0—93之间的整数
        for i in range(len(gene)):
            randnum = np.random.randint(13)
            gene[i][randnum] = 1
        return gene

    # def create_model(self):
    #     # 构建不带分类器的预训练模型
    #     base_model = InceptionV3(weights='imagenet', include_top=False)
    #     #base_model.summary()
    #     # base_model = VGG16(weights='imagenet', include_top=False)#不包含最顶层
    #     # 添加全局平均池化层
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     # 添加一个分类器，假设我们有200个类
    #     predictions = Dense(10, activation='softmax')(x)
    #     # 构建我们需要训练的完整模型
    #     model_type = Model(inputs=base_model.input, outputs=predictions)
    #     return model_type
    def create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False)  # 不包含最顶层
        # 添加全局平均池化层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # 添加一个全连接层
        x = Dense(1024, activation='relu')(x)
        # 添加一个分类器，假设我们有200个类
        predictions = Dense(10, activation='softmax')(x)
        # 构建我们需要训练的完整模型
        model_type = Model(inputs=base_model.input, outputs=predictions)
        return model_type

    def gene_to_phenotype(self, gene_input):
        model = self.create_model()
        list1 = []
        model.layers[-1].trainable = True  # 最后的全连接层可训练
        for j, layer in enumerate(model.layers[:-1]):
            if type(layer) == keras.layers.convolutional.Conv2D:
                list1.append(j)
                index = list1.index(j)
                layer.trainable = gene_input[index]
            if type(layer) != keras.layers.convolutional.Conv2D:  # 非卷积层不可训练
                layer.trainable = False
        #print(list1)
        return model

    def initialization(self):
        #gene = np.random.randint(2, size=(self.pop_size, 22))
        for i in range(self.pop_size):  # 初始定义pop_size=50
            #model = model_type
            model = self.gene_to_phenotype(self.gene[i])
            self.chroms.append(model)
        print('{} network initialization({}) finished.'.format(self.dataset, self.pop_size))

    def evaluation(self, X_train, y_train,X_test, y_test):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            model.compile(loss=self.loss_func, metrics=self.metrics, optimizer='adam')
            # train_loss, train_acc = model.evaluate(_X, _y, verbose=0)
            history = model.fit(X_train, y_train,
                      epochs=25,
                      batch_size=32)
            train_loss, train_acc = history.history['loss'][-1],history.history['accuracy'][-1]
            test_loss, test_acc= model.evaluate(X_test, y_test, batch_size=32)
            cur_evaluation.append({
                'pop': i,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
                'test_loss': round(test_loss, 4),
                'test_acc': round(test_acc, 4),
            })
            print('第{}条染色体对应的基因型为：{}，训练集准确率为：{}，验证集准确率为：{}'.format(i, self.gene[i],history.history['accuracy'][-1],test_acc))
        best_fit = sorted(cur_evaluation, key=lambda x: x['test_acc'])[-1]
        # best_pop =best_fit['pop']  #染色体
        # best_gene=self.gene[best_pop]   #基因型
        self.evaluation_history.append({
            'iter': self.cur_iter + 1,
            'best_fit': best_fit,
            'avg_fitness': np.mean([e['test_acc'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
        })

        print('\nIter: {}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit: {}, avg_fitness: {:.4f}, genetype: {}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness'],self.gene[best_fit['pop']]))

        #保存实验结果
        filename='GA实验结果汇总.txt'
        with open(filename, 'a+') as f:
            f.write('iter轮数\t'.join(self.evaluation_history[-1]['iter']+ '\n'))

            # f.write('iter轮数\tbest_pop\tbest_loss\tbest_acc\tavg_fitness\n')
            for cur in cur_evaluation:
                f.write('第{}条染色体对应的基因型为：{}，训练集准确率为：{}，训练集损失为:{}, 验证集准确率为：{}, 训练集损失为:{},'.format(cur['pop'],
                                        self.gene[cur['pop']],cur['train_acc'],cur['train_loss'], cur['test_acc'], cur['test_loss']+ '\n'))
            f.write('Best_fit: {}, avg_fitness: {:.4f}, genetype: {}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness'],self.gene[best_fit['pop']]))




    def roulette_wheel_selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['test_acc'])#默认升序排序
        cum_acc = np.array([e['test_acc'] for e in sorted_evaluation]).cumsum()#计算累加值
        extra_evaluation = [{'pop': e['pop'], 'test_acc': e['test_acc'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]#返回值[{'pop': 3, 'train_acc': 0.56, 'cum_acc': 0.56},
                                                                            #{'pop': 20, 'train_acc': 0.7, 'cum_acc': 1.26},
                                                                            #{'pop': 2, 'train_acc': 0.8, 'cum_acc': 2.06}
                                                                            #...
                                                                            #{'pop': 1, 'train_acc': 0.99, 'cum_acc': 10.5}]
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
                #break
        return extra_evaluation[-1]['pop']

    @abstractmethod
    def run(self):
        raise NotImplementedError('Run not implemented.')

    @abstractmethod
    def selection(self):
        raise NotImplementedError('Selection not implemented.')

    @abstractmethod
    def crossover(self, _selected_pop):
        raise NotImplementedError('Crossover not implemented.')

    @abstractmethod
    def mutation(self, _selected_pop):
        raise NotImplementedError('Mutation not implemented.')

    @abstractmethod
    def replacement(self, _child):
        raise NotImplementedError('Replacement not implemented.')
