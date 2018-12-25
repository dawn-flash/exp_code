import random
from tool import *
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
import pandas as pd
import json
import os
import sys
parameter = {
    'outDegree':5,
    'pathNum':8,
    'scale':1.0,
    'probesNum':200,
    'k':1000,                 ## 训练数据的次数
    'T_k':100,                ## 测量数据次数
    'data_dir':'/home/zongwangz/文档/Projects/TST/data',
    'save_file':False,
    'e':0.8,                ##判断T值的标准
}
class TST:

    def __init__(self):
        self.num_probe = 200

    @staticmethod
    def genFourTypeData(train_data="train_data", test_data="test_data", scale=1.0,probesNum=200):
        # 一、生成数据
        # 1.生成1000组T=0的训练数据和500组的测试数据
        train_dataNum = 1
        test_dataNum = 1
        R = [1, 2, 3]
        VTree = [4, 4, 4, 0]
        RM = getRM(R,VTree)
        for i in range(1000):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[1]
            dPath = Y[2]
            T = 0
            filename = train_data + '/delay_#' + str(train_dataNum)
            train_dataNum = train_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = train_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        for i in range(500):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[1]
            dPath = Y[2]
            T = 0
            filename = test_data + '/delay_#' + str(test_dataNum)
            test_dataNum = test_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = test_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        # 2.生成1000组T=1的训练数据和500组的测试数据
        R = [1,2,3]
        VTree = [5,5,4,0,4]
        RM = getRM(R, VTree)
        for i in range(1000):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[1]
            dPath = Y[2]
            T = 1
            filename = train_data + '/delay_#' + str(train_dataNum)
            train_dataNum = train_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = train_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        for i in range(500):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[1]
            dPath = Y[2]
            T = 1
            filename = test_data + '/delay_#' + str(test_dataNum)
            test_dataNum = test_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = test_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        # 3.生成1000组T=2的训练数据和500组的测试数据
        for i in range(1000):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[2]
            dPath = Y[1]
            T = 2
            filename = train_data + '/delay_#' + str(train_dataNum)
            train_dataNum = train_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = train_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        for i in range(500):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[2]
            dPath = Y[1]
            T = 2
            filename = test_data + '/delay_#' + str(test_dataNum)
            test_dataNum = test_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = test_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        # 4.生成1000组T=3的训练数据和500组的测试数据
        R = [1,2,3]
        VTree = [4,5,5,0,4]
        RM = getRM(R,VTree)
        for i in range(1000):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[2]
            dPath = Y[1]
            T = 3
            filename = train_data + '/delay_#' + str(train_dataNum)
            train_dataNum = train_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = train_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')
        for i in range(500):
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)
            Y = np.dot(RM, X)
            iPath = Y[0]
            jPath = Y[2]
            dPath = Y[1]
            T = 3
            filename = test_data + '/delay_#' + str(test_dataNum)
            test_dataNum = test_dataNum + 1
            np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
            filename = test_data + '/topo'
            open(filename, "a+").write(str(T) + '\n')

    @staticmethod
    def fourTypesTest(train_data="train_data", test_data="test_data"):
        # 一、生成数据
        # 二、训练模型
        # 1.格式化数据，包括打乱数据
        x_train, y_train = TST.getFormattedDataFromFile(train_data, 4000)
        x_test, y_test = TST.getFormattedDataFromFile(test_data, 2000, False)

        y_train = np_utils.to_categorical(y_train, num_classes=4)
        y_test = np_utils.to_categorical(y_test, num_classes=4)
        '''
        要分别得到四种T值的acc，将test分为四个部分，分别测试得出acc，待完成
        '''
        x_test0 = x_test[0:500]
        x_test1 = x_test[500:1000]
        x_test2 = x_test[1000:1500]
        x_test3 = x_test[1500:2000]
        y_test0 = y_test[0:500]
        y_test1 = y_test[500:1000]
        y_test2 = y_test[1000:1500]
        y_test3 = y_test[1500:2000]
        # 2.获取模型，并且训练，得出loss，acc等
        loss = []
        acc = []
        model = TST.genUnTrainedtModel()
        print('Training ========== (。・`ω´・) ========')
        model.fit(x_train, y_train, epochs=1, batch_size=64)  # 全部训练次数1次，每次训练批次大小64
        if len(x_test0):
            print('Testing ========== (。・`ω´・) ========')
            loss0, accuracy0 = model.evaluate(x_test0, y_test0)  # 测试
            loss.append(loss0)
            acc.append(accuracy0)
            print('\nTest loss0:', loss0)  # 模型偏差
            print('\nTest accuracy0', accuracy0)  # 测试集精度
        if len(x_test1):
            print('Testing ========== (。・`ω´・) ========')
            loss1, accuracy1 = model.evaluate(x_test1, y_test1)  # 测试
            loss.append(loss1)
            acc.append(accuracy1)
            print('\nTest loss1:', loss1)  # 模型偏差
            print('\nTest accuracy1', accuracy1)  # 测试集精度
        if len(x_test2):
            print('Testing ========== (。・`ω´・) ========')
            loss2, accuracy2 = model.evaluate(x_test2, y_test2)  # 测试
            loss.append(loss2)
            acc.append(accuracy2)
            print('\nTest loss2:', loss2)  # 模型偏差
            print('\nTest accuracy2', accuracy2)  # 测试集精度
        if len(x_test3):
            print('Testing ========== (。・`ω´・) ========')
            loss3, accuracy3 = model.evaluate(x_test3, y_test3)  # 测试
            loss.append(loss3)
            acc.append(accuracy3)
            print('\nTest loss3', loss3)  # 模型偏差
            print('\nTest accuracy3', accuracy3)  # 测试集精度
        # model.save_weights('topo1')  # 保存已训练好的模型，‘filename’为保存的文件名

        # 3.画图
        data = {'loss': loss, 'acc': acc}
        frame2 = pd.DataFrame(data, index=['T0', 'T1', 'T2', 'T3'])
        frame2.to_csv('fourTypesTest.csv')
        return loss, acc

    @staticmethod
    def getFormattedDataFromFile(dir_data, delayNum, shuffle=True):
        '''
        从文件里面取出数据格式化 包括打乱
        使用loadtxt函数将delay文件和topo文件中的值分别读到X，Y中
        将X变成四维  shuffle函数打乱  X[index]  Y[index]
        :param: filename 如从filename中取出delay和topo数据
        :param: delayNum  delay文件的个数
        :return: 返回格式化的X,Y
        '''
        for i in range(delayNum):
            filename = dir_data + '/delay_#' + str(i + 1)
            if i == 0:
                x = np.loadtxt(filename, delimiter=',')
            else:
                temp_x = np.loadtxt(filename, delimiter=',')
                x = np.vstack((x, temp_x))
        x = x.reshape(delayNum, 1, 3, parameter['probesNum'])

        filename = dir_data + '/topo'
        y = np.loadtxt(filename)  # 拓扑t 数据

        # 打乱数据
        if (shuffle):
            index = np.arange(delayNum)
            np.random.shuffle(index)
            return x[index], y[index]
        else:
            return x, y

    @staticmethod
    def genUnTrainedtModel():
        model = Sequential()
        model.add(Convolution2D(
            nb_filter=32,  ###第一层卷积层中滤波器的个数#
            nb_row=3,  ###滤波器的长度为5#
            nb_col=3,  ###滤波器的宽度为5#
            border_mode='same',  # padding mode 为same#
            # input_shape=(1, 3, num_probe)
            input_shape=(1, 3, 200)
        ))

        model.add(Activation('relu'))  # 激活函数为relu

        model.add(MaxPooling2D(
            pool_size=(2, 2),  # 下采样格为2*2
            strides=(2, 2),
            padding='same',  # padding mode is 'same'
        ))

        model.add(Convolution2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))

        model.add(Flatten())  # 将多维的输入一维化
        model.add(Dense(1024))  # 全连接层 1024个点
        model.add(Activation('relu'))

        model.add(Dense(4))
        model.add(Activation('softmax'))  # softmax 用于分类

        adam = Adam()  # 学习速率lr=0.0001

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model





    @staticmethod
    def doSimulation():
        tst = TST()  # 生成model
        tst.genD()  #  生成D topo
        # 根据D 和 topo 生成训练数据 然后训练模型
        #生成数据时候 重复的产生端到端数据的次数k
        k = 100
        TST.trainModel(k, tst.D, tst.VTree)
        tst.model = TST.getTrainedModel()
        tst.genM()  # CNN得出M
        E = TST.TST(tst.D.copy(), tst.M.copy())  # TST算法得出E
        # E = TST.numberTree(E, 0)
        print('TST算法推出来的：', E)
        TrueE = TST.VTreetoE(tst.VTree)
        print('真正的E：',TrueE)
        outcome = TST.validateByE(tst.M, TrueE)
        print(outcome[0], outcome[1], outcome[2])


    @staticmethod
    def trainModelOld(k, D, VTree):
        sizeD = len(D)
        # 生成训练数据
        X, Y = TST.genTrainDataOld(k, D, VTree)
        #格式化数据
        num_data = k*sizeD*(sizeD-1)*(sizeD-2)
        num_train = int(2/3*num_data)
        num_test = num_data-num_train

        X = X.reshape(num_train + num_test, 1, 3, 200)
        x_train = X[num_test:, :, :, :]
        x_test = X[:num_test, :, :, :]

        y_train = Y[num_test:(num_train + num_test)]
        y_test = Y[:num_test]
        y_train = np_utils.to_categorical(y_train, num_classes=4)
        y_test = np_utils.to_categorical(y_test, num_classes=4)

        model = TST.geUnTrainedtModel()
        print('Training ========== (。・`ω´・) ========')
        model.fit(x_train, y_train, epochs=1, batch_size=64)  # 全部训练次数1次，每次训练批次大小64
        if len(x_test):
            print('Testing ========== (。・`ω´・) ========')
            loss, accuracy = model.evaluate(x_test, y_test)  # 测试

            print('\nTest loss:', loss)  # 模型偏差
            print('\nTest accuracy', accuracy)  # 测试集精度

        model.save_weights('topo1')  # 保存已训练好的模型，‘filename’为保存的文件名

    @staticmethod
    def trainModel(k, D, VTree):
        sizeD = len(D)
        # 生成训练数据文件
        TST.genTrainData(k, D, VTree,)

        dir_data = "data"
        x, y = TST.getFormattedDataFromFile(dir_data, k*sizeD*(sizeD-1)*(sizeD-2))
        y = np_utils.to_categorical(y, num_classes=4)
        num_data = k * sizeD * (sizeD - 1) * (sizeD - 2)
        num_train = int(2 / 3 * num_data)
        num_test = num_data - num_train
        x_train = x[num_test:]
        x_test = x[:num_test]
        y_train = y[num_test:(num_train + num_test)]
        y_test = y[:num_test]

        model = TST.geUnTrainedtModel()
        print('Training ========== (。・`ω´・) ========')
        model.fit(x_train, y_train, epochs=1, batch_size=64)  # 全部训练次数1次，每次训练批次大小64
        if len(x_test):
            print('Testing ========== (。・`ω´・) ========')
            loss, accuracy = model.evaluate(x_test, y_test)  # 测试

            print('\nTest loss:', loss)  # 模型偏差
            print('\nTest accuracy', accuracy)  # 测试集精度

        model.save_weights('topo1')  # 保存已训练好的模型，‘filename’为保存的文件名



    @staticmethod
    def genTrainDataOld(k, D, VTree):
        E = TST.VTreetoE(VTree)
        X = []
        Y = []
        for i in range(k): #重复k次 k=50
            # 产生链路延迟
            linkDelay = []
            for i in range(VTree.size):
                linkDelay.append(np.random.exponential(1.0, 200))
            # 计算路径延迟
            PathDelay = []  # 对应着D中的顺序
            for i in range(VTree.size):
                if i + 1 not in VTree:
                    tempsum = linkDelay[i]
                    j = i + 1
                    while VTree[j - 1] != 0:
                        j = VTree[j - 1]
                        tempsum = tempsum + linkDelay[j - 1]
                    PathDelay.append(tempsum)

            for dNode in D:
                for iNode in D:
                    if iNode != dNode:
                        for jNode in D:
                            if jNode != dNode and jNode != iNode:
                                T = TST.getCaseValueT(E, iNode, jNode, dNode)
                                # 形成一个delay数据
                                iPath = PathDelay[D.index(iNode)]
                                jPath = PathDelay[D.index(jNode)]
                                dPath = PathDelay[D.index(dNode)]
                                X.append(iPath)
                                X.append(jPath)
                                X.append(dPath)
                                Y.append(T)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    @staticmethod
    def scaleTest():
        scale = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
        ACC = []
        for r in scale:
            loss, acc = TST.fourTypesTest("train_data"+str(r), "test_data"+str(r))
            ACC.append(acc)
        #画图
        # ACC = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.992, 1.0, 1.0, 1.0], [0.998, 0.998, 1.0, 1.0],
        #       [0.992, 0.996, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.984, 0.994, 1.0, 1.0]]
        ACC = np.array(ACC)
        T0 = np.array(ACC[:, 0]).reshape(1, 7)[0]
        T1 = np.array(ACC[:, 1]).reshape(1, 7)[0]
        T2 = np.array(ACC[:, 2]).reshape(1, 7)[0]
        T3 = np.array(ACC[:, 3]).reshape(1, 7)[0]
        fig,ax = plt.subplots()
        plt.xlabel('scale')
        plt.ylabel('acc')
        yticks = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
        ax.set_yticks(yticks)
        ax.set_ylim([0.90, 1.00])
        ax.set_xlim([0.1,1.9])
        plt.plot(scale, T0, 'o-',label='T0')
        plt.plot(scale, T1, 'x-', label='T1')
        plt.plot(scale, T2, '+-', label='T2')
        plt.plot(scale, T3, '.-', label='T3')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0,1), loc=1, borderaxespad=0.)
        plt.show()
    @staticmethod
    def genScaleData():
        scale = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
        import os
        for r in scale:
            os.mkdir("train_data"+str(r))
            os.mkdir("test_data"+str(r))
            TST.genFourTypeData("train_data"+str(r), "test_data"+str(r), r)










    def genD(self, outDegree = 3, pathNum = 5):
        '''
        D = {4,5,6,7,9,10}
        :param outDegree:
        :param pathNum:
        :return:
        '''
        self.VTree = GenTree(outDegree, pathNum)
        nodes = np.array(self.VTree).reshape((1, len(self.VTree)))
        TreePlot(nodes)
        self.D = getLeafNodes(self.VTree)

    def genM(self):
        # 产生链路延迟
        linkDelay = []
        for i in range(self.VTree.size):
            linkDelay.append(np.random.exponential(1.0, self.num_probe))
        # 计算路径延迟
        self.PathDelay = []  # 对应着D中的顺序
        for i in range(self.VTree.size):
            if i + 1 not in self.VTree:
                tempsum = linkDelay[i]
                j = i + 1
                while self.VTree[j - 1] != 0:
                    j = self.VTree[j - 1]
                    tempsum = tempsum + linkDelay[j - 1]
                self.PathDelay.append(tempsum)

        self.M = []
        for dNode in self.D:
            for iNode in self.D:
                if iNode != dNode:
                    for jNode in self.D:
                        if jNode != dNode and jNode != iNode:
                            T = self.getValueTByCNN(iNode, jNode, dNode)
                            self.M.append((iNode, jNode, dNode, T[0]))



    @staticmethod
    def getTrainedModel():
        '''
        获取以及训练好的模型
        :return: model
        '''
        model = Sequential()
        model.add(Convolution2D(
            nb_filter=32,  ###第一层卷积层中滤波器的个数#
            nb_row=3,  ###滤波器的长度为5#
            nb_col=3,  ###滤波器的宽度为5#
            border_mode='same',  # padding mode 为same#
            # input_shape=(1, 3, num_probe)
            input_shape=(1, 3, 200)
        ))

        model.add(Activation('relu'))  # 激活函数为relu

        model.add(MaxPooling2D(
            pool_size=(2, 2),  # 下采样格为2*2
            strides=(2, 2),
            padding='same',  # padding mode is 'same'
        ))

        model.add(Convolution2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))

        model.add(Flatten())  # 将多维的输入一维化
        model.add(Dense(1024))  # 全连接层 1024个点
        model.add(Activation('relu'))

        model.add(Dense(4))
        model.add(Activation('softmax'))  # softmax 用于分类

        adam = Adam()  # 学习速率lr=0.0001

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.load_weights('topo1', by_name=False)
        return model


    @staticmethod
    def TST(D, M):
        '''

        :param D: [4,5,6,7,8]
        :param M: [(4,6,7,3),...]
        :return:E
        '''
        assert isinstance(D, list)
        assert isinstance(M, list)
        E = []
        nodes = []
        for i in range(3):
            index = random.randint(0, len(D)-1)
            nodes.append(D[index])
            del D[index]
        count = 0  # 编号用的

        for t in M:  # 往T里面增加边 严格按照顺序
            if t[0] == nodes[0] and t[1] == nodes[1] and t[2] == nodes[2]:
                if t[3] == 3:
                    E.append((0, 1))
                    E.append((1, nodes[0]))
                    count = 2
                    while count in nodes or count in D:
                        count = count+1
                    E.append((1, count))
                    E.append((count, nodes[2]))
                    E.append((count, nodes[1]))
                    count = count+1
                    break
                if t[3] == 2:
                    E.append((0, 1))
                    count = 2
                    while count in nodes or count in D:
                        count = count + 1
                    E.append((1, count))
                    E.append((1, nodes[1]))
                    E.append((count, nodes[0]))
                    E.append((count, nodes[2]))
                    count = count + 1
                    break
                if t[3] == 1:
                    E.append((0, 1))
                    count = 2
                    while count in nodes or count in D:
                        count = count + 1
                    E.append((1, count))
                    E.append((1, nodes[2]))
                    E.append((count, nodes[0]))
                    E.append((count, nodes[1]))
                    count = count + 1
                    break
                if t[3] == 0:
                    E.append((0, 1))
                    count = 2
                    while count in nodes or count in D:
                        count = count + 1
                    E.append((1, nodes[0]))
                    E.append((1, nodes[1]))
                    E.append((1, nodes[2]))
                    break
        while D:
            index = random.randint(0, len(D)-1)
            dNode = D[index]
            del D[index]
            nodes.append(dNode)


            # obtain v with (2)
            # （1）先在M中找到使T（i，j，d）=0的所有的 i ，j，且i，j出自于已经挑出来的nodes中
            # （2）若为0个,v为空。若只有1个，那么不需要比较公共路径长度，直接找到最近的祖先节点v。如果有多个的话，需要比较公共路径长度
            #    首先在已经形成的T中找到每个（i,j,d）中i，d或者就j，d的共同祖先节点，靠i，d或者j，d最近； 然后
            #   从这些祖先节点中挑出辈分最小的那个，即后代节点（毋庸置疑，所有的祖先节点都在一条路径上）；
            #    最后将此节点作为v

            topo0 = []
            for t in M:
                if t[3] == 0 and t[2] == dNode and t[0] in nodes and t[1] in nodes:
                    topo0.append(t)
            if len(topo0) == 0:
                v = -1 # 为空
            if len(topo0) == 1:
                v = TST.getAncestor(E, topo0[0][0], dNode)
            if len(topo0) > 1:
                # 找到所有的祖先节点
                ancestors = []
                for topo in topo0:
                    ancestor = TST.getAncestor(E, topo[0], topo[1])
                    ancestors.append(ancestor)
                # 比较各个祖先节点  找出辈分最小的 即找公共路径长度最长的
                young = ancestors[0]
                for a in ancestors[1:]:
                    if TST.isDescendant(E, young, a):
                        continue
                    elif TST.isDescendant(E, a, young):
                        young = a
                v = young

                # corollary 1 看在E中是否存在v的子孙叶节点不满足T(i,j,d)!=0 其中i ，j为子孙叶节点
                decendants0 = TST.getDescendants(E, v)
                for t in M:
                    if t[0] in decendants0 and t[1] in decendants0 and t[2] == dNode and t[3] != 0:
                        v = -1
                        break

            if v != -1:
                E.append((v, dNode))
            elif v == -1:
                # obtain u with （3）
                _topo = [] # 先给定义
                topo23 = []
                for t in M:
                    if t[3] > 1 and t[2] == dNode and t[0] in nodes and t[1] in nodes:
                        topo23.append(t)
                if len(topo23) == 0:
                    u = -1  # 为空
                if len(topo23) == 1:
                    u = TST.getAncestor(E, topo23[0][0], topo23[0][1])
                if len(topo23) > 1:
                    # 找到所有的祖先节点
                    ancestors = []
                    Ancestors = []
                    for topo in topo23:
                        ancestor = TST.getAncestor(E, topo[0], topo[1])
                        ancestors.append(ancestor)
                        Ancestors.append((topo[0], topo[1], topo[2], topo[3], ancestor)) # 将拓扑和祖先节点一起存储起来
                    # 比较各个祖先节点  找出辈分最小的 即找公共路径长度最长的
                    young = ancestors[0]
                    for a in ancestors[1:]:
                        if TST.isDescendant(E, young, a):
                            continue
                        elif TST.isDescendant(E, a, young):
                            young = a
                    u = young

                    _topo = [] # 储存祖先节点是u的拓扑
                    for temp in Ancestors:
                        if temp[4] == u:
                            _topo.append((temp[0], temp[1], temp[2], temp[3]))

                if u == -1:
                    u = 0



                k = -1
                tempp = []  # pm，pn的集合
                for t in _topo:
                    if t[3] == 2:  # k=m
                        tempk = t[0]
                        tempp.append(t[0])
                        uDescendant = TST.getDescendants(E, u)
                        flag = True
                        for iNode in uDescendant:
                            if iNode != tempk and (tempk, iNode, dNode, 2) not in M:
                                flag = False
                                break
                        if flag:
                            k = tempk
                            break
                    if t[3] == 3:
                        tempk = t[1]
                        tempp.append(t[1])
                        uDescendant = TST.getDescendants(E, u)
                        flag = True
                        for iNode in uDescendant:
                            if iNode != tempk and (tempk, iNode, dNode, 3) not in M:
                                flag = False
                                break
                        if flag:
                            k = tempk
                            break

                if k == -1:
                    # obtain k with (5)
                    topo1 = []
                    for t in M:
                        if t[3] == 1 and t[2] == dNode and t[0] in nodes and t[1] in nodes:
                            pi = t[0]
                            pj = t[1]
                            f = True
                            for p in tempp:
                                len1 = TST.getSharedPathLenbyNodes(E, p, pi)
                                len2 = TST.getSharedPathLenbyNodes(E, p, pj)
                                len3 = TST.getSharedPathLenbyNode(E, u)
                                if len1 > len3 and len2 > len3:
                                    continue
                                else:
                                    f = False
                            if f:
                                topo1.append(t)
                    if len(topo1) == 0:
                        k = -1  # 为空
                    if len(topo1) == 1:
                        k = TST.getAncestor(E, topo23[0][0], topo23[0][1])
                    if len(topo1) > 1:
                        # 找到所有的祖先节点
                        ancestors = []
                        for topo in topo1:
                            ancestor = TST.getAncestor(E, topo[0], topo[1])
                            ancestors.append(ancestor)
                        # 比较各个祖先节点  找出辈分最大的 即找公共路径长度最短的
                        old = ancestors[0]
                        for a in ancestors[1:]:
                            if TST.isDescendant(E, a, old):
                                continue
                            elif TST.isDescendant(E, old, a):
                                old = a
                        k = old



                # ***********重要假设，输入数据不足，暂时无法测试，只能先假设******************
                # 因为是插入到边上，分两种情况，如果是T>1的情况下，此时，由于u是max所以<u,k>是一条链路，中间不包含节点，
                # 所以要将原来的边给删掉，替换成新的，这个替换应该要在原来边的位置上，不能直接加到后面，不然会对构造树产生影响
                # ，增大编辑距离；如果是T=1的情况下，出现这种情况是 u是尽可能往下找，k是尽可能往上找，找到两点u，k但是,
                # <u,k>中包含了一个节点，即<u,w>,<w,k>,此时将d插入到点w上
                # ******还是有问题********
                while count in nodes or count in D:
                    count = count+1
                if (u, k) in E:
                    insertIndex = E.index((u, k))
                    E.insert(insertIndex, (u, count))
                    E.insert(insertIndex+1, (count, k))
                    E.insert(insertIndex+2, (count, dNode))
                    E.remove((u, k))
                    count = count+1
                else:
                    w = TST.getParent(E, k)
                    if w == -1 or w == 0:
                        print('u=', u, 'k=', k, 'E:', E)
                    insertIndex = E.index((u, w))  # 如果有问题说明不只一个节点
                    E.insert(insertIndex, (w, dNode))


        return E



    @staticmethod
    def validateByE(M, E):
        '''
        验证推断的topo（形式为E）是否正确
        :param M:
        :param E:
        :return: True or False, countFalse, lenM
        '''
        lenM = len(M)
        countTrue = 0
        countFalse = 0
        for t in M:
            T = TST.getCaseValueT(E, t[0], t[1], t[2])
            if T == t[3]:
                countTrue = countTrue+1
            else:
                countFalse = countFalse+1
                print((t[0], t[1], t[2], t[3]),T)
        if countTrue != lenM:
            return False, countFalse, lenM
        elif countTrue == lenM:
            return True, countFalse, lenM

    @staticmethod
    def genCaseData(outDegree, pathNum):
        '''
        生成随机拓扑，根据拓扑生成D，M
        :param outDegree:
        :param pathNum:
        :return: D, M
        '''
        VTree = GenTree(outDegree, pathNum)
        nodes = np.array(VTree).reshape((1, len(VTree)))
        # TreePlot(nodes)
        E = []
        child = 1
        for parent in VTree:
            E.append((parent, child))
            child = child+1
        # 生成D，M
        D = getLeafNodes(VTree)
        M = []
        for dNode in D:
            for iNode in D:
                if iNode != dNode:
                    for jNode in D:
                        if jNode != dNode and jNode != iNode:
                            T = TST.getCaseValueT(E, iNode, jNode, dNode)
                            M.append((iNode, jNode, dNode, T))
        return D, M

    @staticmethod
    def testByCase():
        D, M = TST.genCaseData(4, 8)
        E = TST.TST(D, M)
        outcome = TST.validateByE(M, E)
        print(outcome[0], outcome[1], outcome[2])

    @staticmethod
    def test():
        # D = [2, 4, 6, 7, 8]
        # M = [
        #      (4, 6, 2, 1), (4, 7, 2, 1), (4, 8, 2, 1), (6, 7, 2, 1), (6, 8, 2, 1), (7, 8, 2, 1),
        #      (2, 6, 4, 3), (2, 7, 4, 3), (2, 8, 4, 3), (6, 7, 4, 0), (6, 8, 4, 0), (7, 8, 4, 1),
        #      (2, 4, 6, 3), (2, 7, 6, 3), (2, 8, 6, 3), (4, 7, 6, 0), (4, 8, 6, 0), (7, 8, 6, 1),
        #      (2, 4, 7, 3), (2, 6, 7, 3), (2, 8, 7, 3), (4, 6, 7, 0), (4, 8, 7, 3), (6, 8, 7, 3),
        #      (2, 4, 8, 3), (2, 6, 8, 3), (2, 7, 8, 3), (4, 6, 8, 0), (4, 7, 8, 3), (6, 7, 8, 3),
        #      (6, 4, 2, 1), (7, 4, 2, 1), (8, 4, 2, 1), (7, 6, 2, 1), (8, 6, 2, 1), (8, 7, 2, 1),
        #      (6, 2, 4, 2), (7, 2, 4, 2), (8, 2, 4, 2), (7, 6, 4, 0), (8, 6, 4, 0), (8, 7, 4, 1),
        #      (4, 2, 6, 2), (7, 2, 6, 2), (8, 2, 6, 2), (7, 4, 6, 0), (8, 4, 6, 0), (8, 7, 6, 1),
        #      (4, 2, 7, 2), (6, 2, 7, 2), (8, 2, 7, 2), (6, 4, 7, 0), (8, 4, 7, 2), (8, 6, 7, 2),
        #      (4, 2, 8, 2), (6, 2, 8, 2), (7, 2, 8, 2), (6, 4, 8, 0), (7, 4, 8, 2), (7, 6, 8, 2)
        #      ]
        D = [5, 6, 7, 8, 9, 10]
        M = [
            (6, 7, 5, 1), (7, 6, 5, 1), (6, 8, 5, 3), (8, 6, 5, 2), (6, 9, 5, 3), (9, 6, 5, 2), (6, 10, 5, 3),
            (10, 6, 5, 2), (7, 8, 5, 3), (8, 7, 5, 2), (7, 9, 5, 3), (9, 7, 5, 2), (7, 10, 5, 3), (10, 7, 5, 2),
            (8, 9, 5, 1), (9, 8, 5, 1), (8, 10, 5, 1), (10, 8, 5, 1), (9, 10, 5, 1), (10, 9, 5, 1),
            (5, 7, 6, 3), (7, 5, 6, 2), (5, 8, 6, 1), (8, 5, 6, 1), (5, 9, 6, 1), (9, 5, 6, 1), (5, 10, 6, 1),
            (10, 5, 6, 1), (7, 8, 6, 2), (8, 7, 6, 3), (7, 9, 6, 2), (9, 7, 6, 3), (7, 10, 6, 2), (10, 7, 6, 3),
            (8, 9, 6, 1), (9, 8, 6, 1), (8, 10, 6, 1), (10, 8, 6, 1), (9, 10, 6, 1), (10, 9, 6, 1),
            (5, 6, 7, 3), (6, 5, 7, 2), (5, 8, 7, 1), (8, 5, 7, 1), (5, 9, 7, 1), (9, 5, 7, 1), (5, 10, 7, 1),
            (10, 5, 7, 1), (6, 8, 7, 2), (8, 6, 7, 3), (6, 9, 7, 2), (9, 6, 7, 3), (6, 10, 7, 2), (10, 6, 7, 3),
            (8, 9, 7, 1), (9, 8, 7, 1), (8, 10, 7, 1), (10, 8, 7, 1), (9, 10, 7, 1), (10, 9, 7, 1),
            (5, 6, 8, 2), (6, 5, 8, 3), (5, 7, 8, 2), (7, 5, 8, 3), (5, 9, 8, 3), (9, 5, 8, 2), (5, 10, 8, 3),
            (10, 5, 8, 2), (6, 7, 8, 1), (7, 6, 8, 1), (6, 9, 8, 3), (9, 6, 8, 2), (6, 10, 8, 3), (10, 6, 8, 2),
            (7, 9, 8, 3), (9, 7, 8, 2), (7, 10, 8, 3), (10, 7, 8, 2), (9, 10, 8, 0), (10, 9, 8, 0),
            (5, 6, 9, 2), (6, 5, 9, 3), (5, 7, 9, 2), (7, 5, 9, 3), (5, 8, 9, 3), (8, 5, 9, 2), (5, 10, 9, 3),
            (10, 5, 9, 2), (6, 7, 9, 1), (7, 6, 9, 1), (6, 8, 9, 3), (8, 6, 9, 2), (6, 10, 9, 3), (10, 6, 9, 2),
            (7, 8, 9, 3), (8, 7, 9, 2), (7, 10, 9, 3), (10, 7, 9, 2), (8, 10, 9, 0), (10, 8, 9, 0),
            (5, 6, 10, 2), (6, 5, 10, 3), (5, 7, 10, 2), (7, 5, 10, 3), (5, 8, 10, 3), (8, 5, 10, 2), (5, 9, 10, 3),
            (9, 5, 10, 2), (6, 7, 10, 1), (7, 6, 10, 1), (6, 8, 10, 3), (8, 6, 10, 2), (6, 9, 10, 3), (9, 6, 10, 2),
            (7, 8, 10, 3), (8, 7, 10, 2), (7, 9, 10, 3), (9, 7, 10, 2), (8, 9, 10, 0), (9, 8, 10, 0)
        ]
        E = TST.TST(D, M)
        print(E)
        numberE = TST.numberTree(E)
        _D = TST.getDescendants(E, 0)
        for d in D:
            if d not in _D:
                print('false')
        print("true")
        print(numberE)
        # print(TST.toBracketString(numberE))

    @staticmethod
    def testByCases():
        countAll = 0
        countPass = 0
        outDegreeList = [3, 4, 5]
        pathNumList = [5, 8, 11, 14, 17, 20]
        for outDegree in outDegreeList:
            for pathNum in pathNumList:
                for i in range(10):
                    D, M = TST.genCaseData(outDegree, pathNum)
                    E = TST.TST(D, M)
                    outcome = TST.validateByE(M, E)
                    countAll = countAll+1
                    if outcome[0]:
                        countPass = countPass+1
        print(countPass, countAll)

    @staticmethod
    def doFourTypesTest():
        train_data = '/home/zongwangz/文档/Projects/TST/data/train_data'
        test_data = '/home/zongwangz/文档/Projects/TST/data/test_data'
        # TST.genFourTypeData(train_data,test_data)
        TST.fourTypesTest(train_data, test_data)




    @staticmethod
    def genData(times):
        data = {}
        ## 1.生成topo，生成训练数据
        TST.genTrainData(data,times)
        ## 2.初始化数据
        delayNum = data['delayNum']
        if parameter['save_file']:
            x, y = TST.getFormattedDataFromFile(data['dir_data'],delayNum)
        else:
            x, y = TST.getFormattedDataFromParam(data['x_data'],data['y_data'],delayNum)

        ## 3.训练模型
        model = TST.genUnTrainedtModel()
        x_train = x[:int(0.9*delayNum)]
        y_train = y[:int(0.9*delayNum)]
        x_test = x[int(0.9*delayNum):]
        y_test = y[int(0.9*delayNum):]
        y_train = np_utils.to_categorical(y_train, num_classes=4)
        y_test = np_utils.to_categorical(y_test, num_classes=4)
        model.fit(x_train,y_train)

        if len(x_test):
            print('Testing ========== (。・`ω´・) ========')
            loss0, accuracy0 = model.evaluate(x_test, y_test)  # 测试
            print('\nTest loss0:', loss0)  # 模型偏差
            print('\nTest accuracy0', accuracy0)  # 测试集精度
        data['loss'] = loss0
        data['accuracy'] = accuracy0
        data['model'] = model

        ## 4.产生TST要用的数据
        M = []
        TST.genMData(M,data)
        data['M'] = M
        if len(data['wrongM']) == 0:
            data['wrongRate'] = 0
        else:
            data['wrongRate'] = len(data['wrongM'])/len(M)
        print('VTree:',data['VTree'])
        print('wrongM:',data['wrongM'])
        print('wrongRate:',str(len(data['wrongM']))+'/'+str(len(M)))


        return data


    @staticmethod
    def genTrainData(data, times):
        '''

        :param times:
        :return:
        '''
        #1.获取参数
        outDegree = parameter['outDegree']
        pathNum = parameter['pathNum']
        k = parameter['k']
        scale = parameter['scale']
        probesNum = parameter['probesNum']
        data_dir = parameter['data_dir']
        #2.生成topo
        # VTree0 = GenTree(outDegree=outDegree,pathNum=pathNum)
        # E = numberTopoByVTree(VTree0)
        # VTree = EtoVTree(E)
        VTree = [11, 11, 11, 11, 11, 10, 12, 12, 0, 9, 10, 9]
        E = VTreetoE(VTree)
        R = getLeafNodes(VTree)
        # TreePlot(VTree)
        RM = getRM(R,VTree)
        #3.生成训练数据
        delayNum = 1  #本次实验下的delay文件编号
        dir_data = data_dir+'/alldata/train_data_'+str(times)  #第times次实验的文件夹
        if os.path.exists(dir_data) == False:
            os.mkdir(dir_data)
        x_data = []  ## 存储生成的delay
        y_data = []  ## 存储生成的T值
        for i in range(k):  #重复次数
            X = gen_linkDelay(VTree,scale=scale,probesNum=probesNum)  #链路时延
            Y = np.dot(RM, X)                                           #路径时延
            for dNode in R:
                for iNode in R:
                    if iNode != dNode:
                        for jNode in R:
                            if jNode != dNode and jNode != iNode:
                                #获取T值
                                T = TST.getValueTFromE(E, iNode, jNode, dNode)
                                # 形成一个delay数据
                                iPath = Y[R.index(iNode)]
                                jPath = Y[R.index(jNode)]
                                dPath = Y[R.index(dNode)]
                                # 根据参数来保存文件，或者返回数据
                                if parameter['save_file']:
                                    filename = dir_data + '/delay_#' + str(delayNum)
                                    if os.path.isfile(filename):
                                        os.remove(filename)
                                    np.savetxt(filename, np.array([iPath,jPath,dPath]),fmt='%.6f', delimiter=',')
                                else:
                                    x_data.append(iPath)
                                    x_data.append(jPath)
                                    x_data.append(dPath)
                                    # 保存T值
                                    y_data.append(T)
                                delayNum = delayNum + 1

        #保存topo值
        if parameter['save_file']:
            filename = dir_data + '/topo'
            if os.path.isfile(filename):
                os.remove(filename)
            np.savetxt(filename,y_data,fmt='%d')

        #返回数据
        data['VTree'] = VTree
        data['R'] = R
        data['E'] = E
        data['RM'] = RM
        data['dir_data'] = dir_data  # 第times次文件夹
        data['delayNum'] = delayNum - 1
        if parameter['save_file'] == False:
            data['x_data'] = x_data
            data['y_data'] = y_data
        return x_data,y_data


    @staticmethod
    def getFormattedDataFromParam(x_data,y_data,delayNum,shuffle=True):
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        x = x_data.reshape(delayNum, 1, 3, parameter['probesNum'])
        y = y_data
        # 打乱数据
        if (shuffle):
            index = np.arange(delayNum)
            np.random.shuffle(index)
            return x[index], y[index]
        else:
            return x, y

    @staticmethod
    def genMData(M,data):
        # 获取参数
        VTree = data['VTree']
        E = data['E']
        R = data['R']
        RM = data['RM']
        k = parameter['T_k']
        model = data['model']
        T_dict = {}  #保存T3的多个delay数据cnn推断的T值
        for i in range(k):
            linkDelay = gen_linkDelay(VTree,scale=parameter['scale'],probesNum=parameter['probesNum'])
            pathDelay = np.dot(RM,linkDelay)
            for dNode in R:
                for iNode in R:
                    if iNode != dNode:
                        for jNode in R:
                            if jNode != dNode and jNode != iNode:
                                T = TST.getValueTByCNN(iNode, jNode, dNode,pathDelay,R,model)[0]
                                key = str(iNode)+'_'+str(jNode)+'_'+str(dNode)
                                if key in T_dict:
                                    T_dict[key].append(T)
                                else:
                                    T_dict[key] = [T]
        # 根据执行测量次数k，和CNN推断的T值计算最终T值，和真实的T值比较
        wrongM = [] ## 保存推断错误的T值
        for key in T_dict:
            [s1,s2,s3] = key.split('_')
            iNode = int(s1)
            jNode = int(s2)
            dNode = int(s3)
            inferredT = TST.get_inferredT(T_dict[key])
            trueT = TST.getValueTFromE(E,iNode,jNode,dNode)
            if trueT != inferredT:
                wrongM.append([iNode, jNode, dNode, inferredT])
            M.append([iNode, jNode, dNode, inferredT])
        data['wrongM'] = wrongM


    @staticmethod
    def getValueTByCNN(iNode, jNode, dNode,pathDelay,R,model):
        # 形成一个delay数据
        threePath = pathDelay[R.index(iNode)]
        threePath = np.row_stack((threePath, pathDelay[R.index(jNode)]))
        threePath = np.row_stack((threePath, pathDelay[R.index(dNode)]))
        # 将数据初始化为CNN预测所需要的数据形式
        x = threePath
        x = x[:, 0:parameter['probesNum']]  # 时延 数据
        x = x.reshape(1, 1, 3, parameter['probesNum'])
        x = x[:1, :, :, :]
        # 使用模型预测
        T = model.predict_classes(x)
        return T

    @staticmethod
    def get_inferredT(T_list):
        n = len(T_list)
        if n == 1:
            return T_list[0]
        T0_num = T_list.count(0)/n
        T1_num = T_list.count(1)/n
        T2_num = T_list.count(2)/n
        T3_num = T_list.count(3)/n
        if T0_num >= 0.8:
            return 0
        if T1_num >= 0.8:
            return 1
        if T2_num >= 0.8:
            return 2
        if T3_num >= 0.8:
            return 3
        T = [T0_num,T1_num,T2_num,T3_num]
        ## 这里可以输入这种精度不达标的情况

        return T.index(np.max(T))

    @staticmethod
    def getValueTFromE(E, iNode, jNode, dNode):
        '''
        t(p_i,p_j,p_k) = 0，意味着  p_i ^ p_j = p_i ^ p_d = p_j ^ p_d
        t(p_i,p_j,p_k) = 1，意味着  p_i ^ p_j > p_i ^ p_d = p_j ^ p_d
        t(p_i,p_j,p_k) = 2，意味着  p_i ^ p_j = p_j ^ p_d <  p_i ^ p_d
        t(p_i,p_j,p_k) = 3，意味着  p_i ^ p_j = p_i ^ p_d < p_j ^ p_d

        :param E:
        :param iNode:
        :param jNode:
        :param dNode:
        :return: T=0,1,2,3
        '''
        lenij = getSharedPathLenbyNodes(E, iNode, jNode)
        lenid = getSharedPathLenbyNodes(E, iNode, dNode)
        lenjd = getSharedPathLenbyNodes(E, jNode, dNode)
        T = -1
        if lenij == lenid == lenjd:
            T = 0
        if lenid == lenjd <lenij:
            T = 1
        if lenij == lenjd < lenid:
            T = 2
        if lenij == lenid < lenjd:
            T = 3
        return T
    @staticmethod
    def doSim():
        PC = []  ## 精度
        ED = []  ## 编辑距离
        PN = [_ for _ in range(5, 13)]
        for pathNum in PN:
            cnt = 0
            edit_distance = []
            listObj = []
            for i in range(100):
                log_dir = '/home/zongwangz/文档/Projects/TST/log'
                filename = log_dir+'/console'+str(i)+'.log'
                if os.path.exists(filename):
                    os.remove(filename)
                stdout_backup = sys.stdout
                log_file = open(filename, "w")
                sys.stdout = log_file
                data = TST.genData(i)
                log_file.close()
                sys.stdout = stdout_backup


                TST.TST()


                E = data['E']
                inferredE = data['inferredE']
                ed = calEDbyzss(E, inferredE)
                edit_distance.append(ed)
                if E == inferredE:
                    cnt = cnt + 1
                t1, t2 = toBracketString(E, inferredE)
                dictObj = {
                    "testID": i,
                    "t1": t1,
                    "t2": t2,
                    "d": ed
                }
                listObj.append(dictObj)
            jsonObj = json.dumps(listObj)
            with open('/home/zongwangz/文档/Projects/HTE/data/HTE_' + str(pathNum), 'w') as f:
                f.write(jsonObj)
                f.write('\n')
                f.write("mean edit distance:")
                f.write(str(np.mean(edit_distance)))
                f.write('\n')
                f.write("PC:")
                f.write(str(cnt / 100))
                f.write('\n')
            ED.append(np.mean(edit_distance))
            PC.append(cnt / 100)

        fig1 = plt.subplot()
        plt.xlabel('pathNum')
        plt.ylabel('edit distance')
        plt.plot(PN, ED, 'o-', label='HTE')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

        fig2 = plt.subplot()
        plt.xlabel('pathNum')
        plt.ylabel('PC')
        plt.plot(PN, PC, 'o-', label='HTE')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

    @staticmethod
    def smallTest():
        y1 = []
        y2 =[]
        y3 =[]
        for i in range(100):
            log_dir = '/home/zongwangz/文档/Projects/TST/log'
            filename = log_dir + '/console' + str(i) + '.log'
            if os.path.exists(filename):
                os.remove(filename)
            stdout_backup = sys.stdout
            log_file = open(filename, "w")
            sys.stdout = log_file
            data = TST.genData(i)
            log_file.close()
            sys.stdout = stdout_backup
            y1.append(data['wrongRate'])
            y2.append(data['loss'])
            y3.append(data['accuracy'])
        X = [_ for _ in range(100)]
        fig1 = plt.subplot()
        plt.ylabel('T_wrongRate')
        plt.plot(X, y1, 'o-', label='TST')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

        fig2 = plt.subplot()
        plt.ylabel('loss')
        plt.plot(X, y2, 'o-', label='TST')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

        fig3 = plt.subplot()
        plt.ylabel('accuracy')
        plt.plot(X, y3, 'o-', label='TST')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

if __name__ == "__main__":
    # TST.doSim()
    TST.smallTest()
