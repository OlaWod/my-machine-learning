import numpy as np
import matplotlib.pyplot as plt
from math import exp
import random


def loadDataSet(filename): # 载入数据
    X = []
    y = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        x = [ float(line_arr[i]) for i in range(0, len(line_arr)-1) ]
        x.append(1) # 增广特征向量
        X.append(x)
        y.append(float(line_arr[-1])) # 真实类别
    return X, y


def sigmoid(z):
    return 1/(1+exp(-z))


def gradDescent(X, y): # 梯度下降
    sample_n, trait_n = np.shape(X)
    alpha = 0.1 # 步长
    w = np.ones((trait_n, 1)) # 增广权向量

    max_iter = 900
    for i in range(max_iter):
        wTx = np.array(np.mat(X) * w).flatten()
        p = np.array([sigmoid(zi) for zi in wTx]) # 每个xi的预测值p(wTxi)=p(zi)
        w = w + alpha / sample_n * (np.mat(y - p) * X).T

    return w


def stocGradDescent0(X, y): # 随机梯度下降
    sample_n, trait_n = np.shape(X)
    alpha = 0.1 # 步长
    w = np.ones((trait_n, 1)) # 增广权向量

    for i in range(sample_n):
        wTxi = np.dot(X[i], w)
        p = sigmoid(wTxi) # 该xi的预测值p(wTxi)=p(zi)
        w = w + alpha * (y[i] - p) * np.mat(X[i]).T

    return w


def stocGradDescent1(X, y): # 随机梯度下降
    sample_n, trait_n = np.shape(X)
    w = np.ones((trait_n, 1)) # 增广权向量

    max_iter = 500
    for iter in range(max_iter):
        
        i_no_selected = list(range(sample_n)) # 还未被选择的i们
        for j in range(sample_n):
            alpha = 4 / (1 + iter + j) + 0.0001 # 步长α随迭代次数增加而减小
            i = random.choice(i_no_selected) # 从还没被选的i中选一个i
            
            wTxi = np.dot(X[i], w)
            p = sigmoid(wTxi) # 该xi的预测值p(wTxi)=p(zi)
            w = w + alpha * (y[i] - p) * np.mat(X[i]).T

            i_no_selected.remove(i)

    return w


def plotFig(X, y, w):
    for i in range(len(y)):
        if y[i] > 0: # 1类，蓝色
            plt.plot(X[i][0], X[i][1], 'cp')
        else: # 0类，红色
            plt.plot(X[i][0], X[i][1], 'rp')

    x1 = np.arange(-4, 4, 0.1)
    x2 = (-w[0, 0] * x1 - w[2, 0]) / w[1, 0] # 分离超平面
    plt.plot(x1, x2, 'k')

    plt.xlim((-4,4))
    plt.ylim((-5,20))
    plt.show()


if __name__ == '__main__':
    X, y = loadDataSet('testSet.txt')
    print(X)
    print(y)
    w = stocGradDescent1(X, y)
    print(w)
    plotFig(X, y, w)
    
