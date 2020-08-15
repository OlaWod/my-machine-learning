import numpy as np
import matplotlib.pyplot as plt
from math import exp


def loadDataSet(filename, class_n): # 载入数据
    X = []
    y = []
    fr = open(filename)
    
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        
        x = [ float(line_arr[i]) for i in range(0, len(line_arr)-1) ]
        x.append(1) # 增广特征向量
        X.append(x)

        yt = [0] * class_n
        yt[int(float(line_arr[-1]))] = 1
        y.append(yt) # 真实类别
        
    return X, y


def softmax(W, t, x):
    denominator = 0
    for w in W:
        denominator += exp(np.dot(w, x))
    return exp(np.dot(W[t], x))/denominator


def gradDescent(X, y, class_n): # 梯度下降
    sample_n, trait_n = np.shape(X)
    alpha = 0.1 # 步长
    
    W = []
    for i in range(class_n):
        w = np.ones(trait_n) # 增广权向量
        W.append(w)

    max_iter = 500
    for iter in range(max_iter):
        for t in range(class_n):
            y_minus_soft = np.array([y[i][t] - softmax(W, t, X[i]) for i in range(sample_n)])
            W[t] = W[t] + alpha / sample_n * (np.mat(y_minus_soft) * X)

    return W


def plotFig(X, y, wrong):
    for i in range(len(y)):
        if y[i][0] == 1: # 0类，蓝色
            if(wrong[i]):
                plt.plot(X[i][0], X[i][1], 'bx')
            else:
                plt.plot(X[i][0], X[i][1], 'bp')
        elif y[i][1] == 1: # 1类，红色
            if(wrong[i]):
                plt.plot(X[i][0], X[i][1], 'rx')
            else:
                plt.plot(X[i][0], X[i][1], 'rp')
        else: # 2类，绿色
            if(wrong[i]):
                plt.plot(X[i][0], X[i][1], 'gx')
            else:
                plt.plot(X[i][0], X[i][1], 'gp')
    plt.show()


def predict(x, W, class_n): # 预测x是哪一类
    max_p = 0 # 最大概率
    max_class = -1 # 最可能属于哪一类
    for t in range(class_n):
        p = softmax(W, t, x)
        if(p > max_p):
            max_p = p
            max_class = t
    return max_class


if __name__ == '__main__':
    class_n = 3
    X, y = loadDataSet('testSet.txt', class_n)
    print(X)
    print(y)
    W = gradDescent(X, y, class_n)
    print(W)

    wrong = [0]*len(X)
    for i in range(len(X)):
        if(y[i][predict(X[i], W, class_n)] != 1):
            wrong[i] = 1    
    print('wrong: ' + str(sum(wrong)))
    print('total: ' + str(len(X)))

    plotFig(X, y, wrong)
