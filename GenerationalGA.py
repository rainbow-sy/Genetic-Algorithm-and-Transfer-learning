# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 14:37 
@file: GenerationalGA.py
@author: Matt
"""
import numpy as np
from GA import GA
from keras.layers import Conv2D, Dense
#from keras.models import Sequential, clone_model
import copy
# import testGenerationalGA
import time
class GenerationalGA(GA):
    def run(self):
        print('Generational GA is running...')
        self.initialization()
        while 1:
            #series = self.shuffle_batch()
            t1 = time.time()
            self.evaluation(self.X_train, self.y_train,self.X_test, self.y_test)
            self.selection()
            t2 = time.time()
            # print('第{}轮迭代所需时间{}'.format(self.cur_iter, t2 - t1))
            filename = '迭代时间.txt'
            with open(filename, 'a+') as f:
                f.write('第{}轮迭代所需时间{}'.format(self.cur_iter, t2 - t1) + '\n')
            if self.cur_iter >= self.max_iter:   #如果条件满足停止迭代，跳出函数
                print('Maximum iterations({}) reached.'.format(self.max_iter))
                return
            if self.evaluation_history[-1]['best_fit']['test_acc'] >= self.min_fitness:
                print('Minimum fitness({}) reached.'.format(self.min_fitness))
                return

    def selection(self):
        mating_pool = np.array([self.roulette_wheel_selection() for _ in range(int(self.pop_size/2))])
        np.random.shuffle(mating_pool)
        pairs = []
        pairs.append([mating_pool[0], mating_pool[-1]])
        for i in range(int(self.pop_size/2-1)):
            pairs.append([mating_pool[i], mating_pool[i + 1]])
        pairs = np.array(pairs)
        print('Pairs: {}'.format(list(map(list, pairs))))
        #改变gene,基因型
        children_gene = []
        for pair in pairs:
            children_gene.append(self.crossover(pair))
        print('Cross over finished.')
        children_gene.append(self.reinitial())
        suceed_child = np.array([self.roulette_wheel_selection() for _ in range(int(self.pop_size / 4))])
        #suceed_child_gene=self.gene[suceed_child]
        children_gene.append(self.gene[suceed_child])
        self.replacement(children_gene)
        self.mutation(self.gene)
        #改变chroms，表现型
        del self.chroms[:]
        for i in range(len(self.gene)):
            self.chroms.append(self.gene_to_phenotype(self.gene[i]))

    def crossover(self, _selected_pop):
        # 基因型
        child_gene = np.zeros((1, 13))
        for i in range(self.gene.shape[1]):
            child_gene[0][i] = np.where(self.gene[_selected_pop[0]][i] == self.gene[_selected_pop[1]][i] == 0, 0, 1)
        return child_gene

    def reinitial(self):
        child_gene = self.gene_create(gene_num = int(self.pop_size/4))
        return child_gene

    def mutation(self, mutate_gene):
        for i in range(mutate_gene.shape[0]):
            for j in range(mutate_gene.shape[1]):
                if np.random.rand() < self.p_mutation:
                    mutate_gene[i][j] = 1 - mutate_gene[i][j]
        return mutate_gene

    def replacement(self, _child_gene):
        a = _child_gene[0]
        for i in range(1, len(_child_gene)):
            a = np.vstack((a, _child_gene[i]))
        self.gene[:] = a
        print('Replacement finished.')
        #self.chroms[:] = _child

