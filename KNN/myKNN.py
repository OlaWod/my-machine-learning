import numpy as np
from os import listdir


def img2vector(filename): # 将32*32的图像矩阵转为1*1024的向量
    x = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            x[0, 32*i+j] = int(line[j])
    return x


def classify(X, y, x, k=3): # KNN分类
    diff_mat_sq = (np.tile(x, (X.shape[0], 1)) - X)**2
    dist = (diff_mat_sq.sum(axis=1))**0.5 # x与X中每个样本的欧式距离
    sorted_dist_idx = dist.argsort() # dist升序排序的索引

    label_count = {}
    for i in range(k):
        label = y[sorted_dist_idx[i]] # 距离最近的第i个样本的标签
        label_count[label] = label_count.get(label, 0) + 1 # 计数

    result = -1
    count = -1
    for label in label_count:
        if(label_count[label] > count): # result是k个最近样本中最多的那个标签
            result = label
            count = label_count[label]

    return result
        

def handwritingClassTest():
    filelist_train = listdir('trainingDigits') # 训练文件名list
    X = np.zeros((len(filelist_train), 1024))
    y = []

    for i in range(len(filelist_train)):
        filename = filelist_train[i] # 文件名
        X[i] = img2vector('trainingDigits/%s' % filename) # 将该文件的图片转为向量
        label = int(filename.split('.')[0].split('_')[0]) # 该文件对应的标签
        y.append(label)
        
    filelist_test = listdir('testDigits') # 测试文件名list
    error_n = 0
    
    for i in range(len(filelist_test)):
        filename = filelist_test[i] # 文件名
        x = img2vector('testDigits/%s' % filename)
        label = int(filename.split('.')[0].split('_')[0])
        result = classify(X, y, x)
        print('result: %d, label:%d' % (result, label))
        if(result != label):
            error_n += 1

    print('error num: %d' % error_n)
    print('error rate: %f' % (error_n/len(filelist_test)))


if __name__ == '__main__':
    handwritingClassTest()
