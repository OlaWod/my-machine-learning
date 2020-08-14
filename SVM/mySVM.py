import numpy as np
import random
import math
import matplotlib.pyplot as plt


def loadDataSet(file_name): # 载入数据
    X = []
    y = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        X.append( [ float(line_arr[i]) for i in range(0, len(line_arr)-1) ] )
        y.append( float(line_arr[-1]) )
    return X, y


class SVM:
    def __init__(self, max_iter = 40, C = 0.6, toler = 0.001, kernel_type = 'linear'):
        self.max_iter = max_iter
        self.C = C
        self.toler = toler
        self.kernels = {
            'linear' : self.linearKernel,
            'quadratic' : self.quadraticKernel,
            'polynomial' : self.polynomialKernel,
            'gaussian' : self.gaussianKernel
            }
        self.kernel = self.kernels[kernel_type]

        
    def fit(self, X, y):
        sample_n, trait_n = np.shape(X)
        alpha = np.zeros(sample_n)
        b = 0

        iter = 0
        while(iter < self.max_iter):
            alpha_changed = False
            
            for i in range(sample_n): # 选择x1 = xi
                
                Ei = self.calEk(alpha, X, y, i, b)
                
                # 如果X[i]这个样本点违反KKT条件的程度大于toler，就继续选择x2
                if((y[i]*Ei < -self.toler and alpha[i] < self.C) or (y[i]*Ei > self.toler and alpha[i] >0)):
                    
                    j = self.selectJ(sample_n, i) # 选择x2 = xj
                    Ej = self.calEk(alpha, X, y, j, b)

                    eta = self.kernel(X[i], X[i]) + self.kernel(X[j], X[j]) - 2*self.kernel(X[i], X[j])
                    if(eta == 0):
                        continue

                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    alpha[j] = alpha_j_old + y[j]*(Ei - Ej)/eta
                    
                    L, H = self.calLH(y[i], y[j], alpha_i_old, alpha_j_old) # 计算αj的上下界
                    alpha[j] = self.clipAlpha(alpha[j], L, H) # 调整αj使其在范围内
                    if(abs(alpha[j] - alpha_j_old) < 0.00001):
                        continue

                    bi = self.calB(b, X[i], X[j], y[i], y[j], alpha[i], alpha_i_old, alpha[j], alpha_j_old, Ei, X[i])
                    bj = self.calB(b, X[i], X[j], y[i], y[j], alpha[i], alpha_i_old, alpha[j], alpha_j_old, Ej, X[j])
                    if(alpha[i] > 0 and alpha[i] < self.C):
                        b = bi
                    elif(alpha[j] > 0 and alpha[j] < self.C):
                        b = bj
                    else:
                        b = (bi + bj)/2

                    alpha_changed = True
                    print("iter: %d, i:%d, j:%d, alpha[i]:%f, alpha[j]:%f" % (iter, i, j, alpha[i], alpha[j]))

            if(alpha_changed):
                iter = 0
            else:
                print(iter)
                iter += 1 # iter是alpha没发生更新的次数
        
        self.alpha = alpha
        self.w = self.calW(alpha, X, y) # 非线性可分的时候这个没用
        self.b = b
        self.X = X
        self.y = y

                                  
    def selectJ(self, sample_n, i): # 随机选一个和i不一样的j
        j = i
        while(j==i):
            j = random.randint(0, sample_n-1) # [0, sample_n-1]
        return j


    def clipAlpha(self, alpha_j, L, H): # 调整αj使其在范围内
        if(alpha_j < L):
            alpha_j = L
        if(alpha_j > H):
            alpha_j = H
        return alpha_j


    def calLH(self, yi, yj, alpha_i, alpha_j): # 计算αj的上下界
        if(yi != yj):
            return max(0, alpha_j - alpha_i), min(self.C, self.C + alpha_j - alpha_i)
        else:
            return max(0, alpha_j + alpha_i - self.C), min(self.C, alpha_j + alpha_i)

        
    def calEk(self, alpha, X, y, k, b):
        K = [self.kernel(X[k], X[i]) for i in range(len(y))]
        return np.dot(np.multiply(alpha, y), K) + b - y[k]
    

    def calB(self, b, x1, x2, y1, y2, alpha_1, alpha_1_old, alpha_2, alpha_2_old, Ek, xk):
        return b - Ek - y1*self.kernel(x1, xk)*(alpha_1-alpha_1_old) - y2*self.kernel(x2, xk)*(alpha_2-alpha_2_old)


    def calW(self, alpha, X, y):
        return np.mat(np.multiply(alpha, y)) * np.mat(X)


    def linearKernel(self, x1, x2): # 线性核
        return np.dot(x1, x2)


    def quadraticKernel(self, x1, x2): # 平方核
        return np.dot(x1, x2)**2


    def polynomialKernel(self, x1, x2, p = 1): # 多项式核
        return (np.dot(x1, x2) + 1)**p


    def gaussianKernel(self, x1, x2, sigma = 1): # 高斯核
        return math.exp(- (np.linalg.norm(x1-x2)**2) / (2*sigma**2))


    def predict(self, x): # 判断新点x属于哪一类
        K = [self.kernel(x, self.X[i]) for i in range(len(self.y))]
        fx = np.dot(np.multiply(self.alpha, self.y), K) + self.b
        print(fx)
        if(fx > 0):
            return 1
        else:
            return -1


def plotFig(X, y, w, b, alpha): # 绘制图像（仅限样例数据集testSet.txt）
    for i in range(len(y)):
        if y[i] > 0: # +1类，蓝色
            if alpha[i] > 0:
                plt.plot(X[i][0], X[i][1], 'cp')
            else:
                plt.plot(X[i][0], X[i][1], 'cx')
        else: # -1类，红色
            if alpha[i] > 0:
                plt.plot(X[i][0], X[i][1], 'rp')
            else:
                plt.plot(X[i][0], X[i][1], 'rx')

    x1 = np.arange(-1, 10, 0.1)
    x2 = (-w[0, 0] * x1 - b) / w[0, 1] # 分离超平面
    x2_1 = (1 - w[0, 0] * x1 - b) / w[0, 1] # 间隔边界
    x2_2 = (-1 - w[0, 0] * x1 - b) / w[0, 1] # 间隔边界
    plt.plot(x1, x2, 'k')
    plt.plot(x1, x2_1, 'm')
    plt.plot(x1, x2_2, 'm')

    plt.xlim((-1,10))
    plt.ylim((-7,5))
    plt.show()
    

if __name__ == '__main__':
    X, y = loadDataSet('testSet.txt')
    print(X)
    print(y)
    model = SVM()
    model.fit(X, y)
    print(model.alpha)
    print(model.w)
    print(model.b)
    x = [5, 6]
    print(model.predict(x))
    plotFig(X, y, model.w, model.b, model.alpha)
    
