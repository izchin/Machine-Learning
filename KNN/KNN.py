from numpy import *
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import struct


def ReadLabel(fileName):
    fileHandle = open(fileName, 'rb')   # 以二进制打开文件
    fileContent = fileHandle.read()     # 读取到缓冲区中
    head = struct.unpack_from('>II', fileContent, 0)    # 取前两个整数 ，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]     # label数
    # print(labelNum)
    bitstring = '>' + str(labelNum) + 'B'   # fmt格式：'>47040000B'
    label = struct.unpack_from(bitstring, fileContent, offset)      # 取data数据，返回一个元组
    return np.array(label)

def ReadImage(fileName):
    fileHandle = open(fileName, "rb")
    fileContent = fileHandle.read()
    offset = 0
    head = struct.unpack_from('>IIII', fileContent, offset)
    offset += struct.calcsize('>IIII')
    imageNum = head[1]
    rows = head[2]
    cols = head[3]
    images = np.empty((imageNum, 784))
    imageSize = rows * cols
    fmt = '>' + str(imageSize) + 'B'
    for i in range(imageNum):
        images[i] = np.array(struct.unpack_from(fmt, fileContent, offset))
        offset += struct.calcsize(fmt)
    return images


def KNN(testData, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    distance1 = tile(testData, dataSetSize).reshape((60000, 784)) - dataSet
    distance2 = distance1**2
    distance3 = distance2.sum(axis=1)
    distance4 = distance3**0.5
    # 欧式距离计算结束
    sortedDistIndicies = distance4.argsort()    # 返回从小到大排序的索引
    classCount = np.zeros(10, np.int32)   # 10个类别
    # 统计前k个数据类的数量
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] += 1
    max = 0
    id = 0
    print(classCount.shape[0])
    for i in range(classCount.shape[0]):
        if classCount[i] >= max:
            max = classCount[i]
            id = i
    print(id)
    return id


# 文件获取和测试
def TestKNN():
    # 文件读取
    # minst数据集
    # 训练集文件
    train_image = 'MINST_DATA/train-images.idx3-ubyte'
    # 测试集文件
    test_image = 'MINST_DATA/t10k-images.idx3-ubyte'
    # 训练集标签文件
    train_label = 'MINST_DATA/train-labels.idx1-ubyte'
    # 测试集标签文件
    test_label = 'MINST_DATA/t10k-labels.idx1-ubyte'

    # 读取数据
    trainImage = ReadImage(train_image)
    testImage = ReadImage(test_image)
    trainLabel = ReadLabel(train_label)
    testLabel = ReadLabel(test_label)

    testRatio = 0.01       # 取数据集的前0.01为测试数据
    trainRow = trainImage.shape[0]
    testRow = testImage.shape[0]
    testNum = int(testRow * testRatio)
    errorCount = 0      # 判断错误的个数
    for i in range(testNum):
        result = KNN(testImage[i], trainImage, trainLabel, 30)

        print(result, testLabel[i])
        if result != testLabel[i]:
            errorCount += 1.0
    errorRate = errorCount / float(testNum)
    acc = 1.0 - errorRate
    print(errorCount)
    print("\nthe total number of errors is :%d" % errorCount)
    print("\nthe total error rate is :%f" % errorRate)
    print("\nthe total accuracy  rate is :%f" % acc)


if __name__ == "__main__":
    TestKNN()
