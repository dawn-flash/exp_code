# -*- coding: utf-8 -*-
from tool import *
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model
'''
@project:exp_code
@author:zongwangz
@time:19-2-19 下午8:43
@email:zongwang.zhang@outlook.com
'''
'''
本代码是在更新了诸多函数后，对TST代码的更新和测试
'''
def TST(R,M):
    '''
    使用TST算法进行拓扑推断
    :param R: 目标节点集
    :param M: 三路子拓扑结构信息集
    :return: InferredE
    '''

    D = R.copy()
    E = []
    nodes = []
    ##选择三个目标节点作为初始的拓扑结构
    nodes.extend([1,2,3])
    D.remove(1)
    D.remove(2)
    D.remove(3)

    count = len(R)+1  # 编号用的

    for t in M:  # 往T里面增加边 严格按照顺序
        if t[0] == nodes[0] and t[1] == nodes[1] and t[2] == nodes[2]:
            if t[3] == 3:
                E.append((0, count))
                E.append((count, nodes[0]))
                E.append((count, count+1))
                count+=1
                E.append((count, nodes[2]))
                E.append((count, nodes[1]))
                count = count + 1
                break
            if t[3] == 2:
                E.append((0, count))
                E.append((count, count+1))
                E.append((count, nodes[1]))
                count+=1
                E.append((count, nodes[0]))
                E.append((count, nodes[2]))
                count = count + 1
                break
            if t[3] == 1:
                E.append((0, count))
                E.append((count, count+1))
                E.append((count, nodes[2]))
                count+=1
                E.append((count, nodes[0]))
                E.append((count, nodes[1]))
                count = count + 1
                break
            if t[3] == 0:
                E.append((0, count))
                E.append((count, nodes[0]))
                E.append((count, nodes[1]))
                E.append((count, nodes[2]))
                count+=1
                break
    while D:
        ##顺序选择点加入 把随机该为顺序可以大幅度降低编辑距离和提高PC
        dNode = D[0]
        del D[0]
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
            v = -1  # 为空
        if len(topo0) == 1:
            v = getAncestor(E, topo0[0][0], dNode)
        if len(topo0) > 1:
            # 找到所有的祖先节点
            ancestors = []
            for topo in topo0:
                ancestor = getAncestor(E, topo[0], topo[1])
                ancestors.append(ancestor)
            # 比较各个祖先节点  找出辈分最小的 即找公共路径长度最长的
            young = ancestors[0]
            for a in ancestors[1:]:
                if isDescendant(E, young, a):
                    continue
                elif isDescendant(E, a, young):
                    young = a
            v = young

            # corollary 1 看在E中是否存在v的子孙叶节点不满足T(i,j,d)!=0 其中i ，j为子孙叶节点
            decendants0 = getDescendants(E, v)
            for t in M:
                if t[0] in decendants0 and t[1] in decendants0 and t[2] == dNode and t[3] != 0:
                    v = -1
                    break

        if v != -1:
            E.append((v, dNode))
        elif v == -1:
            # obtain u with （3）
            _topo = []  # 先给定义
            topo23 = []
            for t in M:
                if t[3] > 1 and t[2] == dNode and t[0] in nodes and t[1] in nodes:
                    topo23.append(t)
            if len(topo23) == 0:
                u = -1  # 为空
            if len(topo23) == 1:
                u = getAncestor(E, topo23[0][0], topo23[0][1])
            if len(topo23) > 1:
                # 找到所有的祖先节点
                ancestors = []
                Ancestors = []
                for topo in topo23:
                    ancestor = getAncestor(E, topo[0], topo[1])
                    ancestors.append(ancestor)
                    Ancestors.append((topo[0], topo[1], topo[2], topo[3], ancestor))  # 将拓扑和祖先节点一起存储起来
                # 比较各个祖先节点  找出辈分最小的 即找公共路径长度最长的
                young = ancestors[0]
                for a in ancestors[1:]:
                    if isDescendant(E, young, a):
                        continue
                    elif isDescendant(E, a, young):
                        young = a
                u = young

                _topo = []  # 储存祖先节点是u的拓扑
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
                    uDescendant = getDescendants(E, u)
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
                    uDescendant = getDescendants(E, u)
                    flag = True
                    for iNode in uDescendant:
                        if iNode != tempk and (iNode, tempk, dNode, 3) not in M:
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
                            len1 = getSharedPathLenbyNodes(E, p, pi)
                            len2 = getSharedPathLenbyNodes(E, p, pj)
                            len3 = getSharedPathLenbyNode(E, u)
                            if len1 > len3 and len2 > len3:
                                continue
                            else:
                                f = False
                        if f:
                            topo1.append(t)
                if len(topo1) == 0:
                    k = -1  # 为空
                if len(topo1) == 1:
                    k = getAncestor(E, topo23[0][0], topo23[0][1])
                if len(topo1) > 1:
                    # 找到所有的祖先节点
                    ancestors = []
                    for topo in topo1:
                        ancestor = getAncestor(E, topo[0], topo[1])
                        ancestors.append(ancestor)
                    # 比较各个祖先节点  找出辈分最大的 即找公共路径长度最短的
                    old = ancestors[0]
                    for a in ancestors[1:]:
                        if isDescendant(E, a, old):
                            continue
                        elif isDescendant(E, old, a):
                            old = a
                    k = old

            if (u, k) in E:
                insertIndex = E.index((u, k))
                E.insert(insertIndex, (u, count))
                E.insert(insertIndex + 1, (count, k))
                E.insert(insertIndex + 2, (count, dNode))
                E.remove((u, k))
                count = count + 1
            else:
                w = getParent(E, k)
                if w == -1 or w == 0:
                    print('u=', u, 'k=', k, 'E:', E)
                insertIndex = E.index((u, w))  # 如果有问题说明不只一个节点
                E.insert(insertIndex, (w, dNode))

    # return numberTopoByVTree(EtoVTree(E))
    return numberTopo(E,R)

def for_test_TST():
    '''
    对TST算法的测试，假设输入中M全部为正确的M
    :return:
    '''
    pc = []
    ed = []
    VTrees = getVTrees("/home/zongwangz/exp_code/Topo_4_3_10")
    serial_number = 0
    for VTree in VTrees:
        R = getLeafNodes(VTree)
        E = numberTopoByVTree(VTree) ##这一步存在问题，输入的VTree必须为第一版的VTree，否则转换会出现问题
        M = []
        for iNode in R:
            for jNode in R:
                for dNode in R:
                    if iNode != jNode and jNode != dNode and iNode != dNode:
                        T = getValueTFromE(E,iNode,jNode,dNode)
                        M.append((iNode,jNode,dNode,T))
        inferredE = TST(R,M)
        print(serial_number,":")
        print("原始topo：",E)
        print("推断拓扑：",inferredE)
        if inferredE == E:
            pc.append(1)
            ed.append(0)
        else:
            pc.append(0)
            ed.append(calEDbyzss(E,inferredE))
        serial_number+=1

    PN = [_ for _ in range(3, 11)]
    PC = [np.mean(pc[i*100:(i+1)*100]) for i in range(8)]
    ED = [np.mean(ed[i*100:(i+1)*100]) for i in range(8)]

    ##画图
    fig1 = plt.subplot()
    plt.xlabel('pathNum')
    plt.ylabel('edit distance')
    plt.plot(PN, ED, 'o-', label='TST')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

    fig2 = plt.subplot()
    plt.xlabel('pathNum')
    plt.ylabel('PC')
    plt.plot(PN, PC, 'o-', label='TST')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

def getValueTFromE(E, iNode, jNode, dNode):
    '''
    用于test_TST算法，获取真实的T值
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
    if lenid == lenjd < lenij:
        T = 1
    if lenij == lenjd < lenid:
        T = 2
    if lenij == lenid < lenjd:
        T = 3
    return T
def genTopo(pathNumList=[i+3 for i in range(8)],num_VTree=100,outDegree=3):
    '''
    产生本代码需要使用的拓扑，其中保存的VTree是第一版的VTree形式 [0,1,1,2,2]
    :return:
    '''
    filename = "/home/zongwangz/exp_code/ns-3/TST/topo" + "_" + str(outDegree) + "_" + str(pathNumList[0]) + "_" + str(pathNumList[-1])
    for pathNum in pathNumList:
        for i in range(num_VTree):
            VTree = GenTree(outDegree, pathNum)
            with open(filename, 'a+') as f:
                f.write(str(list(VTree)))
                f.write('\n')

def getTopo(filename="/home/zongwangz/exp_code/ns-3/TST/topo_3_3_10"):
    '''
    从topo_3_3_10中获取所有的VTree
    :return:
    '''
    VTrees = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            VTrees.append(ast.literal_eval(line))
    return VTrees

def doSim(dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data",train_data="/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data/train_data", test_data="/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data/test_data",probesNum=500):
    '''
    完整的仿真流程
    :return:
    '''
    #1.产生数据
    if not os.path.exists(train_data+"/topo"):
        genData(train_data,test_data,probesNum)
    #2.训练模型
    if not os.path.exists(dir+"/model.h5"):
        trainModel(dir,train_data,test_data,probesNum)
    #3.CNN推断三路子拓扑信息
    #(1)产生拓扑和数据
    if not os.path.exists(dir+"/simData/VTree"):
        genVTree(dir+"/simData")
    VTrees = getVTrees(dir+"/simData/VTree")
    if not os.path.exists(dir+"/simData/input/"+"README"):
        genCNNInput(dir+"/simData",VTrees)
    #(2)CNN 推断
    if not os.path.exists(dir+"/simData/output/README"):
        doCNNInfer(dir,len(VTrees))
    #删除上一次推断的E

    #4.算法推断
    if not os.path.exists(dir+"/simData/inferredE"):
        doAlgInfer(dir,VTrees)

    #5.检测CNN推断三路子拓扑和算法推断结果
    print_CNN_result(dir+"/simData",VTrees)
    print_alg_result(dir+"/simData")

def print_alg_result(dir):
    filename1 = dir+"/E"
    filename2 = dir+"/inferredE"
    E = open(filename1,"r").readlines()
    inferredE = open(filename2,"r").readlines()
    l = int(len(E)/100)
    result = [0 for i in range(l)]
    pathNum = [i+3 for i in range(l)]
    for i in range(len(E)):
        index = int(i/100)
        if E[i] == inferredE[i]:
            result[index] += 1
    for i in range(l):
        result[i] = result[i]/100

    np.savetxt(dir + "/result2", result)
    fig = plt.subplot()
    plt.xlabel(u"路径数量")
    plt.ylabel(u"推断正确率")
    plt.plot(pathNum,result,".-",label = u"算法推断准确率")
    plt.title(u"TST算法推断(500)")
    plt.legend()
    plt.savefig(dir+"/result2")
    plt.show()
    plt.close()


def print_CNN_result(dir,VTrees):
    input_dir = dir+"/input"
    output_dir = dir+"/output"
    l = int(len(VTrees)/100)
    pathNum = [i+3 for i in range(l)]
    good = [0 for i in range(l)]
    bad = [0 for i in range(l)]
    result0 = [0 for i in range(l)] ##保存拓扑推断失败率
    for i in range(len(VTrees)):
        index = int(i/100)
        filename1 = input_dir+"/vtree"+str(i+1)+"/vtree"+str(i+1)
        trueM = np.loadtxt(filename1,int)
        filename2 = output_dir+"/M"+str(i+1)
        inferredM = np.loadtxt(filename2,int)
        flag = True #表示这个拓扑CNN完全推断正确
        for j in range(len(trueM)):
            if (trueM[j] == inferredM[j]).all():
                good[index] += 1
            else:
                flag = False
                bad[index] += 1
        if flag == False:
            result0 [index] += 1
    for i in range(l):
        result0[i]=result0[i] / 100
    result = [0 for i in range(l)] #保存三路子拓扑推断成功率
    for i in range(len(result)):
        result[i] = good[i]/(good[i]+bad[i])
    np.savetxt(dir+"/result1",result)
    np.savetxt(dir+"/result0",result0)

    fig = plt.subplot()
    plt.xlabel(u"路径数量")
    plt.ylabel(u"概率")
    plt.plot(pathNum,result,".-",label = u"三路子拓扑推断正确率")
    plt.plot(pathNum,result0,"o-",label = u"拓扑推断错误率")
    plt.title(u"训练好的CNN推断效果(500)")
    plt.legend()
    plt.savefig(dir+"/result1")
    plt.show()
    plt.close()


def doAlgInfer(dir,VTrees):
    if os.path.exists(dir + "/simData/inferredE"):
        os.remove(dir + "/simData/inferredE")
    for i in range(len(VTrees)):
        R = getLeafNodes(VTrees[i])
        # trueM = np.loadtxt(dir+"/simData/input/vtree"+str(i+1)+"/vtree"+str(i+1),int)
        M = np.loadtxt(dir+"/simData/output/M"+str(i+1),int)
        ##输入M是list TST处理的是元组，需要转换一下
        dotM = []
        for item in M:
            dotM.append(tuple(item))
        inferredE = []
        try:
            inferredE = TST(R, dotM)
        except BaseException:
            # print("推断出错")
            inferredE = []
        open(dir + "/simData/inferredE", "a+").write(str(inferredE))
        open(dir + "/simData/inferredE", "a+").write("\n")


def doCNNInfer(dir,num):
    '''
    使用训练好的CNN模型来推断三路子拓扑
    :param 所有数据的根目录
    :param num VTrees的大小
    :return:
    '''
    model = load_model(dir + "/model.h5")
    for i in range(num):
        input_dir = dir+"/simData/input/vtree"+str(i+1)
        trueM = np.loadtxt(input_dir+"/vtree"+str(i+1))
        for j in range(len(trueM)):
            filename = input_dir + '/delay_#' + str(j + 1)
            if j == 0:
                x = np.loadtxt(filename, delimiter=',')
            else:
                temp_x = np.loadtxt(filename, delimiter=',')
                x = np.vstack((x, temp_x))
        x = x.reshape(len(trueM), 1, 3, 1)
        y = model.predict_classes(x)
        output_M = dir+"/simData/output/M"+str(i+1)
        M = copy.deepcopy(trueM)
        for k in range(len(M)):
            M[k][3] = y[k]
        np.savetxt(output_M,M,"%.0f")

        ##检测推断结果
        acc = 0
        print(i + 1)
        for k in range(len(trueM)):
            if trueM[k][3] == y[k]:
                acc += 1
        print(acc, "/", len(trueM))
    open(dir+"/simData/output/README","w").write("Done")


def genCNNInput(dir,VTrees):
    if os.path.exists(dir+"/input/README"):
        os.remove(dir+"/input/README")
    directory_number = 1
    for VTree in VTrees:
        delay_number = 1
        #生成文件夹存储一个topo的时延数据
        if not os.path.exists(dir+"/input/vtree"+str(directory_number)):
            os.mkdir(dir+"/input/vtree"+str(directory_number))
        #生成链路时延和路径时延
        linkDelay = gen_linkDelay(VTree,500)
        R = getLeafNodes(VTree)
        RM = getRM(R,VTree)
        pathDelay = np.dot(RM,linkDelay)
        #生成三路子拓扑时延数据
        R = getLeafNodes(VTree)
        E = VTreetoE(VTree)
        M = []
        for iNode in R:
            for jNode in R:
                for dNode in R:
                    if iNode != jNode and jNode != dNode and iNode != dNode:
                        T = getValueTFromE(E, iNode, jNode, dNode)
                        M.append((iNode, jNode, dNode, T))
                        filename = dir + "/input/vtree" + str(directory_number) + '/delay_#' + str(delay_number)
                        np.savetxt(filename, np.array([Covariance_way2(pathDelay[iNode-1], pathDelay[jNode-1]), Covariance_way2(pathDelay[iNode-1], pathDelay[dNode-1]),
                                                  Covariance_way2(pathDelay[jNode-1], pathDelay[dNode-1])]), fmt='%.6f', delimiter=',')
                        delay_number+=1
        filename1 = dir+"/input/vtree"+str(directory_number)+'/vtree' +str(directory_number)
        directory_number += 1
        np.savetxt(filename1,M,'%.0f')
    open(dir+"/input/README",'w').write("Done")


def genVTree(dir):
    '''
    产生100个四路子拓扑 VTree
    :return:
    '''
    MaxOutDegree = 3
    PathNum = 4
    filename = dir+"/VTree"
    filename1 = dir+"/E"
    VTrees = []
    for c in range(100):
        VTree0 = GenTree(MaxOutDegree, PathNum)
        E = numberTopoByVTree(VTree0)
        VTree = EtoVTree(E)
        VTrees.append(VTree)
        with open(filename, 'a+') as f:
            f.write(str(VTree))
            f.write('\n')
        with open(filename1, "a+") as f:
            f.write(str(E))
            f.write('\n')

def trainModel(dir,train_data,test_data,probesNum,train_data_num=80000,test_data_num=20000):
    # 1.格式化数据，包括打乱数据
    x_train, y_train = getFormattedDataFromFile(train_data, train_data_num, probesNum)
    x_test, y_test = getFormattedDataFromFile(test_data,test_data_num,probesNum, False)

    y_train = np_utils.to_categorical(y_train, num_classes=4)
    y_test = np_utils.to_categorical(y_test, num_classes=4)
    '''
    要分别得到四种T值的acc，将test分为四个部分，分别测试得出acc，待完成
    '''
    x_test0 = x_test[0:int(test_data_num/4)]
    x_test1 = x_test[int(test_data_num/4):int(test_data_num*2/4)]
    x_test2 = x_test[int(test_data_num*2/4):int(test_data_num*3/4)]
    x_test3 = x_test[int(test_data_num*3/4):int(test_data_num*4/4)]
    y_test0 = y_test[0:int(test_data_num/4)]
    y_test1 = y_test[int(test_data_num/4):int(test_data_num*2/4)]
    y_test2 = y_test[int(test_data_num*2/4):int(test_data_num*3/4)]
    y_test3 = y_test[int(test_data_num*3/4):int(test_data_num*4/4)]
    # 2.获取模型，并且训练，得出loss，acc等
    loss = []
    acc = []
    model = genUnTrainedtModel()
    print('Training ==================')
    model.fit(x_train, y_train, epochs=1, batch_size=100)  # 全部训练次数1次，每次训练批次大小64
    if len(x_test0):
        print('Testing ==================')
        loss0, accuracy0 = model.evaluate(x_test0, y_test0)  # 测试
        loss.append(loss0)
        acc.append(accuracy0)
        print('\nTest loss0:', loss0)  # 模型偏差
        print('\nTest accuracy0', accuracy0)  # 测试集精度
    if len(x_test1):
        print('Testing ==================')
        loss1, accuracy1 = model.evaluate(x_test1, y_test1)  # 测试
        loss.append(loss1)
        acc.append(accuracy1)
        print('\nTest loss1:', loss1)  # 模型偏差
        print('\nTest accuracy1', accuracy1)  # 测试集精度
    if len(x_test2):
        print('Testing ==================')
        loss2, accuracy2 = model.evaluate(x_test2, y_test2)  # 测试
        loss.append(loss2)
        acc.append(accuracy2)
        print('\nTest loss2:', loss2)  # 模型偏差
        print('\nTest accuracy2', accuracy2)  # 测试集精度
    if len(x_test3):
        print('Testing ==================')
        loss3, accuracy3 = model.evaluate(x_test3, y_test3)  # 测试
        loss.append(loss3)
        acc.append(accuracy3)
        print('\nTest loss3', loss3)  # 模型偏差
        print('\nTest accuracy3', accuracy3)  # 测试集精度

    model.save(dir+'/model.h5')
    # model.save_weights(dir+'/model')  # 保存已训练好的模型，‘filename’为保存的文件名
    return loss, acc

def genData(train_data="/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data/train_data", test_data="/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data/test_data",probesNum=500):
    '''
    产生数据
    尝试使用基本的三路子拓扑数据训练，不过每条链路的时延都不一样
    :return:
    '''
    # 一、生成数据
    # 1.生成20000组T=0的训练数据和5000组的测试数据
    train_dataNum = 1
    test_dataNum = 1
    R = [1, 2, 3]
    VTree = [4, 4, 4, 0]
    RM = getRM(R, VTree)
    for i in range(20000):
        X = gen_linkDelay(VTree,probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[1]
        dPath = Y[2]
        T = 0
        filename = train_data + '/delay_#' + str(train_dataNum)
        train_dataNum = train_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array([Covariance_way2(iPath,jPath), Covariance_way2(iPath,dPath), Covariance_way2(jPath,dPath)]), fmt='%.6f', delimiter=',')
        filename = train_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    for i in range(5000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[1]
        dPath = Y[2]
        T = 0
        filename = test_data + '/delay_#' + str(test_dataNum)
        test_dataNum = test_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = test_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    # 2.生成1000组T=1的训练数据和500组的测试数据
    R = [1, 2, 3]
    VTree = [5, 5, 4, 0, 4]
    RM = getRM(R, VTree)
    for i in range(20000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[1]
        dPath = Y[2]
        T = 1
        filename = train_data + '/delay_#' + str(train_dataNum)
        train_dataNum = train_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = train_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    for i in range(5000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[1]
        dPath = Y[2]
        T = 1
        filename = test_data + '/delay_#' + str(test_dataNum)
        test_dataNum = test_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = test_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    # 3.生成1000组T=2的训练数据和500组的测试数据
    for i in range(20000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[2]
        dPath = Y[1]
        T = 2
        filename = train_data + '/delay_#' + str(train_dataNum)
        train_dataNum = train_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = train_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    for i in range(5000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[2]
        dPath = Y[1]
        T = 2
        filename = test_data + '/delay_#' + str(test_dataNum)
        test_dataNum = test_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = test_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    # 4.生成1000组T=3的训练数据和500组的测试数据
    R = [1, 2, 3]
    VTree = [4, 5, 5, 0, 4]
    RM = getRM(R, VTree)
    for i in range(20000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[2]
        dPath = Y[1]
        T = 3
        filename = train_data + '/delay_#' + str(train_dataNum)
        train_dataNum = train_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = train_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')
    for i in range(5000):
        X = gen_linkDelay(VTree, probesNum=probesNum)
        Y = np.dot(RM, X)
        iPath = Y[0]
        jPath = Y[2]
        dPath = Y[1]
        T = 3
        filename = test_data + '/delay_#' + str(test_dataNum)
        test_dataNum = test_dataNum + 1
        # np.savetxt(filename, np.array([iPath, jPath, dPath]), fmt='%.6f', delimiter=',')
        np.savetxt(filename, np.array(
            [Covariance_way2(iPath, jPath), Covariance_way2(iPath, dPath), Covariance_way2(jPath, dPath)]), fmt='%.6f',
                   delimiter=',')
        filename = test_data + '/topo'
        open(filename, "a+").write(str(T) + '\n')

def gen_linkDelay(VTree,probesNum = 500):
    linkDelay = []
    for i in range(len(VTree)):
        # loc = np.random.randint(1,10)
        # scale = 0.5*loc
        # linkDelay.append(np.random.normal(loc,scale,probesNum))
        scale = np.random.uniform(1,2.5)
        linkDelay.append(np.random.exponential(scale,1000))
    return linkDelay

def getFormattedDataFromFile(dir_data, delayNum, probesNum=500,shuffle=True):
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
    x = x.reshape(delayNum, 1, 3, 1)

    filename = dir_data + '/topo'
    y = np.loadtxt(filename)  # 拓扑t 数据

    # 打乱数据
    if (shuffle):
        index = np.arange(delayNum)
        np.random.shuffle(index)
        return x[index], y[index]
    else:
        return x, y

def genUnTrainedtModel():
    model = Sequential()
    model.add(Convolution2D(
        nb_filter=32,  ###第一层卷积层中滤波器的个数#
        nb_row=3,  ###滤波器的长度为5#
        nb_col=3,  ###滤波器的宽度为5#
        border_mode='same',  # padding mode 为same#
        # input_shape=(1, 3, num_probe)
        input_shape=(1, 3, 1)
    ))
    print(model.get_output_shape_at(0))
    model.add(Activation('relu'))  # 激活函数为relu

    model.add(MaxPooling2D(
        pool_size=(2, 2),  # 下采样格为2*2
        strides=(2, 2),
        padding='same',  # padding mode is 'same'
    ))
    print(model.get_output_shape_at(0))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    print(model.get_output_shape_at(0))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(2, 2), padding='same'))
    print(model.get_output_shape_at(0))
    model.add(Flatten())  # 将多维的输入一维化
    model.add(Dense(1024))  # 全连接层 1024个点
    print(model.get_output_shape_at(0))
    model.add(Activation('relu'))

    model.add(Dense(4))
    model.add(Activation('softmax'))  # softmax 用于分类
    print(model.get_output_shape_at(0))
    adam = Adam()  # 学习速率lr=0.0001


    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
if __name__ == "__main__":
    doSim()
#     pathNum = [3,4,5,6,7,8,9,10]
#     result1 = [9.599999999999999645e-01,
# 8.499999999999999778e-01,
# 6.700000000000000400e-01,
# 4.799999999999999822e-01,
# 3.599999999999999867e-01,
# 2.300000000000000100e-01,
# 1.100000000000000006e-01,
# 4.000000000000000083e-02,]
#     result2 = [9.899999999999999911e-01,
# 9.300000000000000488e-01,
# 8.800000000000000044e-01,
# 7.099999999999999645e-01,
# 4.000000000000000222e-01,
# 4.099999999999999756e-01,
# 2.899999999999999800e-01,
# 1.499999999999999944e-01]
#     result3 = [9.899999999999999911e-01,
# 9.899999999999999911e-01,
# 9.499999999999999556e-01,
# 8.800000000000000044e-01,
# 9.100000000000000311e-01,
# 7.600000000000000089e-01,
# 6.300000000000000044e-01,
# 5.100000000000000089e-01]
#     result4 = [1,1,1,1,1,1,1,1]
#     fig = plt.subplot()
#     plt.plot(pathNum,result1,".-",color = "red",label = "200")
#     plt.plot(pathNum, result2,"o-", color="blue", label="500")
#     plt.plot(pathNum, result3,"x-" ,color="black", label="1000")
#     plt.plot(pathNum, result4, "*-",color="yellow", label="输入全部正确")
#     plt.title("算法推断精度")
#     plt.xlabel("路径数目")
#     plt.ylabel("准确率")
#     plt.legend()
#     plt.savefig("/media/zongwangz/RealPAN-13438811621/myUbuntu/TST_data/result")
#     plt.show()
#     plt.close()