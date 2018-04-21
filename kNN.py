# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2018-03-01 18:28:47
# @Last Modified by:   Marte
# @Last Modified time: 2018-03-10 20:39:53
#科学计算包Numpy
from numpy import *

import numpy as np
#运算符模块
import operator

def classify(inX, dataSet, labels, k):
    '''
    定义KNN算法分类器函数
    inX: 测试数据
    dataSet: 训练数据
    labels: 分类类别
    k: k值
    '''

    dataSetSize = dataSet.shape[0] #shape[0]获取一维长度
    diffMat = np.tile(inX,(dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2 # **求幂
    sqDistances = sqDiffMat.sum(axis=1) #sum(axis=1): 矩阵行向量相加
    distances = sqDistances **0.5 #欧氏距离
    sortedDistIndicies = distances.argsort() #排序并返回index(索引)

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

def classify_two(inX, dataSet, labels, k):
    m, n = dataSet.shape
    #计算测试数据到每个点的欧氏距离
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5) #统计所有与测试数据的距离

    sortDist = sorted(distances) #对所有与测试数据距离的数组排序

    #k个最近的值所属的类别
    classCount = {}
    for i in range(k):
        voteLabel = labels[distances.index(sortDist[i])] #获取到距离测试数据进的训练数据label
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 #统计label出现的次数加入到classCount中
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True) #根据label出现的次数进行排序
    return sortedClass[0][0] #返回出现次数最多的label

def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# make a script both importable and executable
if __name__ == '__main__':
    dataSet, labels = createDataSet()
    r = classify_two([0,0.2], dataSet, labels, 3)
    print(r)


