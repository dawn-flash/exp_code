'''
2018.11.02 19:40
author zzw

做ns3仿真用的工具python文件.和ns3_tool功能不同，这个主要是用来处理一整个流程。
1.分析trace文件，得到metric，保存为中间文件
2.更加通用形式的算法
3.得出精度和编辑距离，画图

2018.11.02 20:10
给出最基本的函数
'''
import re
import copy
import numpy as np
from tool import *
import os
import shutil
import time
from sklearn.mixture import GaussianMixture as GMM
import networkx as nx
import json
from scipy.stats import t as T_d
from scipy.stats import norm
import scipy.stats
parameter = {
    'K':200,
    'Norm':20,
    'Nij':10,
    'probesNum':200,
}

def getLinkDelayWithB2B(parentNode,childNode,srcIP,destIPList,filename):
    '''
    返回某条链路上的时延，B2B 下返回背靠背包的时延，考虑丢包情况
    :param parentNode:
    :param childNode:
    :param srcIP:
    :param destIPList:
    :param filename:
    :return:
    '''
    start_time = {}
    end_time = {}
    linkdelay = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ", line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            currentNode = int(namespace.split("/")[2]) ##当前所处的节点
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            if oc[39] == 'ns3::UdpHeader':
                src_port = int(oc[43])
                dest_port = int(oc[45])
                if src_ip == srcIP and dest_ip in destIPList:  ##由此确定是背靠背中的包
                    key = dest_ip+str(dest_port)+str(packet_id)  ##唯一一个包
                    if action == '+' and currentNode == parentNode and key not in start_time:
                        start_time[key] = time
                    if action == 'r' and currentNode == childNode and key in start_time:
                        end_time[key] = time
    for id in end_time:
        packet_duration = end_time[id]-start_time[id]
        linkdelay.append(packet_duration)
    return linkdelay

def getPathDelayWithB2B(srcIP,srcNode,destIP,destPort,destNode,filename):
    '''
    2018.11.02 20：10
    从一串背靠背中获取一条路径的时延
    需要参数如下，源IP，源节点编号，目的IP ，目的端口，目的节点编号，文件名称
    默认UDP 包，代码的解析格式是UDP
    需要考虑丢包的状况
    :return:
    '''
    highest_packet_id = 0  ##记录最大的包ID
    start_time = {}    ##记录发包时间
    end_time = {}       ## 记录到达时间
    rec_id = []         ## 记录接收的id
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ", line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            currentNode = int(namespace.split("/")[2]) ##当前所处的节点
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            if oc[39] == 'ns3::UdpHeader':
                dest_port = int(oc[45])
                if src_ip == srcIP and dest_ip == destIP and dest_port == destPort:  ##可以唯一确定一串包
                    if packet_id > highest_packet_id:
                        highest_packet_id = packet_id
                    if action == '+' and currentNode == srcNode:
                        start_time[packet_id] = time
                    if action == 'r' and currentNode == destNode:
                        end_time[packet_id] = time
                        rec_id.append(packet_id)
    delay = []
    delaySum = 0
    for id in rec_id:
        if id in start_time and id in end_time:
            packet_duration = end_time[id] - start_time[id]  ##一个包的延迟
            delaySum = packet_duration + delaySum  ##所有包的延迟
            delay.append(packet_duration)
    data = {
        "mean_delay": delaySum / len(rec_id),
        "delay": delay,
        "rec_id": rec_id,
    }
    return data


'''
算法函数
'''
def getRNJThreshold(VTree,filename):
    '''
    公共路径上最小长度的一半，这个长度可以使用经过这条链路的包的　时延差的方差来表示。
    :return:
    '''
    srcIP = ""   ##
    destIPList = [] ##
    E = VTreetoE(VTree)
    internalNodes = getInternalNodes(VTree)
    linkLen = []
    for childNode in internalNodes:
        parentNode = getParent(E,childNode)
        delay = getLinkDelayWithB2B(parentNode,childNode,srcIP,destIPList,filename)
        length = Variance_way1(delay)
        linkLen.append(length)
    return np.min(linkLen)/2
def RNJ(R,S,e):
    '''
    RNJ的算法函数，参数为目地节点，默认源节点为0，ｍｅｔｒｉｃ的矩阵，和阈值ｅ
    问题主要在于阈值ｅ的获得上，ｅ取最小链路长度的一半
    :param data:
    :return:
    '''
    hatR = R.copy()
    dotR = R.copy()
    V = [0]
    E = []
    n = len(dotR)
    number = n + 1
    while len(dotR) != 1:
        for node in R:
            if node in R and node not in dotR:
                for i in range(len(S[0])):
                    S[R.index(node)][i] = S[i][R.index(node)] = 0
        indexs = np.where(np.max(S) == S)[0]
        iIndex = indexs[0]
        jIndex = indexs[1]
        R.append(number)
        dotR.remove(R[iIndex])
        dotR.remove(R[jIndex])
        V.append(R[iIndex])
        V.append(R[jIndex])
        E.append((number, R[iIndex]))
        E.append((number, R[jIndex]))
        brother = []
        for kNode in dotR:
            kIndex = R.index(kNode)
            if S[iIndex][jIndex] - S[iIndex][kIndex] < e:
                brother.append(kNode)
                V.append(kNode)
                E.append((number, kNode))
        for node in brother:
            dotR.remove(node)
        n = len(R)
        tempS = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(n - 1):
                tempS[i][j] = S[i][j]
        S = tempS
        for node in dotR:
            index = R.index(node)
            S[n - 1][index] = S[index][n - 1] = S[iIndex][index]
        dotR.append(number)
        number = number + 1
    E.append((0, dotR[0]))
    inferredE = numberTopo(E, hatR)
    return inferredE

'''
HTE
'''
def HTE(S,R):
    '''

    :param pathNum:
    :param outDegree:
    :return:
    '''

    # init
    V = R.copy()  ##新节点的集合
    V.append(0)
    I = []  ##每一轮 待处理的内节点集合
    Finish = False  ## 结束标志
    N = len(R) + 1
    V.append(N)
    newE = [(0, N)]  ##  new edge set
    for k in R:
        newE.append((N, k))
    I.append(N)
    N = N + 1

    while not Finish:
        dotI = []
        Finish = True
        for interNode in I:
            children = getChildren(newE, interNode)

            X = []  ##GMM 输入数据
            keyDict = []  ##  记录本次处理的目的节点
            children.sort()
            for iNode in children:
                for jNode in children:
                    if jNode > iNode:
                        key = "S" + str(iNode) + "," + str(jNode)
                        keyDict.append(key)
                        Tij = S[key]
                        for tij in Tij:
                            X.append([tij])
            X = np.array(X)

            #  根据BIC准则得到component的个数
            BIC = np.infty
            for k in range(1, 4):
                if len(X) < k:
                    continue
                gmm = GMM(n_components=k, covariance_type='spherical')
                gmm.fit(X)
                bic = gmm.bic(X)

                if bic < BIC:
                    BIC = bic
                    best_gmm = gmm
            # print("components:", best_gmm.n_components)
            # if best_gmm.n_components == 1:
            #     continue
            gmm = best_gmm
            score1 = gmm.score_samples(X)
            score2 = np.sum(score1)
            predict = gmm.predict(X)
            means = np.mean(gmm.means_, 1)
            ## 从Ketdict里面得到  待处理的  目标节点dest_node
            dest_node = []
            for key in keyDict:
                str1, str2 = key.rsplit(',')
                str1 = str1[1:]
                node1 = int(str1)
                node2 = int(str2)
                if node1 not in dest_node:
                    dest_node.append(node1)
                if node2 not in dest_node:
                    dest_node.append(node2)
            # 先构建key_data,是Tij中Nij个元素的标签
            n_samples = len(predict)
            Nij = int(parameter['Nij'])
            key_num = n_samples / Nij
            key_data = {}
            tempc = 1
            dest_node.sort()
            for inode in dest_node:
                for jnode in dest_node:
                    if jnode > inode:
                        key_data['T' + str(inode) + ',' + str(jnode)] = predict[(tempc - 1) * Nij:tempc * Nij]
                        tempc = tempc + 1

            ## progressive search alg  找到最小cluster数目的clusters
            Kp, Bp = progressive_search_Alg(means, dest_node, key_data)

            if len(Kp[0]) == 1:
                ## 经过psa合并后 只剩下一个簇，不需要经过HCS 而且此时的means差距其实比较小
                continue

            ##构建HCS的输入数据,得到 hatK，hatB,Kp不只一个，将HCS的结果的似然函数结果保存在Lp中
            Lp = []
            for i in range(len(Kp)):
                Ki = Kp[i]
                Bi = Bp[i]
                VDict = []
                for j in range(len(Ki)):
                    VDict.append(j)
                weightedE = []
                for node1 in VDict:
                    for node2 in VDict:
                        if node2 > node1:
                            w = calWOnSets(Ki[node1], Ki[node2], key_data, Bi)
                            weightedE.append((node1, node2, w))
                ##HCS
                G = nx.Graph()
                G.add_weighted_edges_from(weightedE)
                subgraph_list = []
                div_factor = 2
                HCS(G, subgraph_list, div_factor)
                # 得到HCS的结果 clusters
                clusters = []
                for subgraph in subgraph_list:
                    cluster = []
                    for item in subgraph:
                        cluster.extend(Ki[VDict[item]])
                    clusters.append(cluster)

                # 计算似然函数值
                score3 = calL(clusters, S, gmm)
                Lp.append((score3, clusters, Bi))
            ## 选取似然值最大的
            if len(Lp) == 1:
                hatL = Lp[0][0]
                hatK = Lp[0][1]
                hatB = Lp[0][2]
            else:
                lp = []
                for item in Lp:
                    lp.append(item[0])
                index = lp.index(np.max(lp))
                hatL = Lp[index][0]
                hatK = Lp[index][1]
                hatB = Lp[index][2]
            ## postmerge Alg
            bestK = []
            bestScore = hatL
            if len(hatK) > 1:
                betterResult = []
                postmerge_Alg(hatK, hatL, hatB, key_data, S, gmm, betterResult)
                if len(betterResult) != 0:
                    for result in betterResult:
                        if result[0] > bestScore:
                            bestScore = result[0]
                            bestK = result[1]
                    hatL = bestScore
                    hatK = bestK

            if len(hatK) > 1:
                for A in hatK:
                    if len(A) > 1:
                        V.append(N)
                        for j in A:
                            if (interNode, j) in newE:
                                newE.remove((interNode, j))
                                if (interNode, N) not in newE:
                                    newE.append((interNode, N))
                                if (N, j) not in newE:
                                    newE.append((N, j))
                        if len(A) > 2:
                            dotI.append(N)
                            Finish = False
                        N = N + 1
        I = dotI
    inferredE = numberTopo(newE, R)
    # TreePlot(EtoVTree(inferredE))
    return inferredE

def progressive_search_Alg(means,dest_node,key_data):
    '''
    依次合并component，并preclustering，选出最少clusters的分类Kp，可能多个Bi对应的cluster的数目
    最少，所以Kp长度不唯一。
    计算W的时候，i，j均为单个节点，计算的函数为calW（）
    :param gmm:
    :param X:
    :param keyDict:
    :return:
    '''
    ## progressive search alg

    # 构建排序好的 分量 orderB，将分量的均值从小到大排列
    orderB = {}
    temp_means = copy.copy(means).tolist()
    for i in range(len(means)):
        min_mean = np.min(temp_means)
        assert isinstance(temp_means, list)
        temp_means.remove(min_mean)
        index = np.where(means == min_mean)[0][0]
        orderB[str(i + 1)] = (min_mean, index)
    # 分别构造B1，B2，B3等 precluser的输入
    B_dict = {}
    for i in range(len(orderB)):
        B_dict['B' + str(i + 1)] = [str(j + 1) for j in range(i + 1)]
    # precluster K_dict 对应着B_dict的输出
    K_dict = {}
    for i in range(len(B_dict)):
        K_dict['K'+str(i+1)] = precluster_Alg(dest_node, key_data, B_dict['B'+str(i+1)],orderB)

    # 从K_dict里面选出cluster数量最少的那个K
    min_num = np.infty
    Kp = []
    Bp = []
    #获取最小数目min_num
    for key in K_dict:
        K = K_dict[key]
        c_nun = len(K)
        if c_nun < min_num:
            min_num = c_nun
    #获取对应min_num的cluster Kp 和 Bp
    for key in K_dict:
        K = K_dict[key]
        if min_num == len(K):
            Kp.append(K)
            label = []
            for c in B_dict['B'+key.partition('K')[2]]:
               label.append(orderB[c][1])
            Bp.append(label)
    return Kp,Bp
def calWOnSets(A1, A2, key_data, B):
    '''
    eq(7)的计算函数，计算两个cluster在B下的 权值
    :param A1:
    :param A2:
    :param key_data:
    :param B:
    :return:
    '''
    lenA1 = len(A1)
    lenA2 = len(A2)
    Nij = parameter['Nij']
    n_sum = lenA1*lenA2*Nij
    N = 0
    for node1 in A1:
        for node2 in A2:
            if node1 > node2:
                key = 'T'+str(node2)+','+str(node1)
            else:
                key = 'T'+str(node1)+','+str(node2)
            Tij = key_data[key]
            for label in Tij:
                if label in B:
                    N = N+1

    W = 1-N/n_sum
    return W
def calL(clusters,S,gmm):
    '''
    计算cluster的似然值，在从HCS得到的结果中挑选最好的，在postmerge中要使用到
    :param clusters:
    :param S:
    :param gmm:
    :return:
    '''
    # 创建似然函数的数据
    cluster_num = len(clusters)
    l_data = []
    if cluster_num == 1:
        for node_1 in clusters[0]:
            for node_2 in clusters[0]:
                if node_2 == node_1:
                    continue
                if node_1 > node_2:
                    key = 'S' + str(node_2) + ',' + str(node_1)
                else:
                    key = 'S' + str(node_1) + ',' + str(node_2)
                T_ij = S[key]
                for gamaij in T_ij:
                    l_data.append([gamaij])
    else:
        for i in range(cluster_num):
            for j in range(cluster_num):
                if j > i:
                    for node_1 in clusters[i]:
                        for node_2 in clusters[j]:
                            if node_1 > node_2:
                                key = 'S' + str(node_2) + ',' + str(node_1)
                            else:
                                key = 'S' + str(node_1) + ',' + str(node_2)
                            T_ij = S[key]
                            for gamaij in T_ij:
                                l_data.append([gamaij])
    score1 = gmm.score_samples(l_data)
    score0 = sum(score1)
    score2 = np.mean(score1)
    return  score0
def postmerge_Alg(hatK,hatL,hatB,key_data,S,gmm,betterResult):
    '''
    pairwise merge 选择似然值最大的
    :param hatK:
    :param hatL:
    :return:
    '''
    merge_conditions = []  ## 对于hatK 里面的 每一个 cluster 都有一个 pairwise merge
    for cluster in hatK:
        merge_condition = []
        #选取权值最大的连接
        weight = 0
        sc_cluster = []
        for other_cluster in hatK:
            if cluster != other_cluster:
                w = calWOnSets(cluster,other_cluster,key_data,hatB)
                if w > weight:
                    weight = w
                    sc_cluster = other_cluster
        tempcluster = []
        tempcluster.extend(cluster)
        tempcluster.extend(sc_cluster)
        tempcluster.sort()
        merge_condition.append(tempcluster)
        for item in hatK:
            if item != cluster and item != sc_cluster:
                merge_condition.append(item)
        if merge_condition not in merge_conditions:
            merge_conditions.append(merge_condition)
    # 计算每一个merge_condition下的似然值，与原似然值比较，若大于原似然值，则继续pairwise merge
    # 否则，接受原hatK的分类
    for dotK in merge_conditions:
        score = calL(dotK,S,gmm)
        if score > hatL:
            betterResult.append([score,dotK])
            postmerge_Alg(dotK,score,hatB,key_data,S,gmm,betterResult)
def precluster_Alg(dest_node, key_data, B, orderB):
    '''
    将dest_node分类
    :param dest_node:
    :param predict:
    :param B:
    :param orderB
    :return: clusters
    '''
    #先把B换成对应的标签，在order中有记录
    label = []
    for item in B:
        label.append(orderB[item][1])
    #对于dest_node 中的每一个点 做出F的集合
    F = {}
    for iNode in dest_node:
        F[str(iNode)] = []
        for jNode in dest_node:
            if jNode != iNode:
                #构造Key 保持顺序
                if jNode > iNode:
                    key = 'T'+str(iNode)+','+str(jNode)
                else:
                    key = 'T'+str(jNode)+','+str(iNode)
                #获取计算Weigt的数据
                Tij = key_data[key]
                W = calW(Tij, label)
                if W < 1/2:
                    F[str(iNode)].append(jNode)
    #将F中的元素分类，当F(i) = F(j) 则i，j在一个cluster中
    K = []
    for node in dest_node:
        if str(node) in F:
            tempK = [node]
            for key in F:
                if key != str(node):
                    temp1 = F[str(node)]
                    temp2 = F[key]
                    temp1.sort()
                    temp2.sort()
                    if temp1 == temp2 :
                        tempK.append(int(key))
            for node in tempK:
                del F[str(node)]
            K.append(tempK)
    return K
def calW(Tij, label):
    '''
    判断所给的Tij是否属于对应的lable，计算Weight，i，j均为单个节点
    :param Tij:
    :param B:
    :return:
    '''
    Nij = parameter['Nij']
    N = 0
    for gama in Tij:
        if gama in label:
            N = N+1
    W = 1-N/Nij
    return W


'''
T-test
'''


def TP(E,R,S):
    dotV = [0]
    child = getChildren(E,0)[0]
    inferredE = []
    inferredE.append((0,child))
    dotV.append(child)
    U = [child]
    while len(U) != 0:
        j = U[0]
        U.remove(j)
        U.extend(getChildren(E, j))
        if j not in R and isZeroLink(E,j,S):
            children = getChildren(E, j)
            parent = getParent(inferredE, j)
            for k in children:
                inferredE.append((parent, k))
                if (parent, j) in inferredE:
                    inferredE.remove((parent, j))
        else:
            children = getChildren(E, j)
            for k in children:
                inferredE.append((j, k))
                dotV.append(k)
    inferredE = numberTopo(inferredE, R)
    return inferredE

def isZeroLink(E, child, S):
    '''
    判断当前链路是否应该被剪去
    :param E:
    :param child:
    :param S:
    :return:
    '''
    Tchild = []
    Tparent = []
    parent = getParent(E,child)
    if parent == 0:
        return  False
    descendant1 = getDescendants(E,parent)
    descendant1.sort()
    for iNode in descendant1:
        for jNode in descendant1:
            if jNode > iNode:
                if getAncestor(E,iNode,jNode) == parent:
                    key = "S"+str(iNode)+","+str(jNode)
                    Tparent.extend(S[key])
    descendant2 = getDescendants(E, child)
    descendant2.sort()
    for iNode in descendant2:
        for jNode in descendant2:
            if jNode > iNode:
                if getAncestor(E, iNode, jNode) == child:
                    key = "S"+str(iNode) + "," + str(jNode)
                    Tchild.extend(S[key])
    Tparent = np.array(Tparent)
    Tchild = np.array(Tchild)
    return t_test(Tparent, Tchild)

def t_test(X1,X2,alpha = 0.005):
    '''
    使用T-test 的方法判别两组数据均值的差异性
    :param X1:
    :param X2:
    :param alpha:
    :return:
    '''
    hatX1 = np.mean(np.array(X1))
    hatX2 = np.mean(np.array(X2))
    squareX1 = Variance_way1(X1)
    squareX2 = Variance_way1(X2)
    n1 = len(X1)
    n2 = len(X2)
    t = (hatX2 - hatX1) / np.sqrt(squareX1 / n1 + squareX2 / n2)
    df1 = np.square(squareX1 / n1 + squareX2 / n2)
    df2 = np.square(squareX1 / n1) / (n1) + np.square(squareX2 / n2) / (n2)
    df = df1 / df2
    interval = T_d.interval(1 - 2 * alpha, df)  ## two tails
    if t > interval[1]:
        return False
    else:
        return True
'''
ALT
'''


def ALT(starS, R):
    S = [[0 for _ in range(len(R))] for _ in range(len(R))]
    for iNode in R:
        for jNode in R:
            if jNode > iNode:
                key = "S"+str(iNode)+","+str(jNode)
                S[R.index(iNode)][R.index(jNode)] = S[R.index(jNode)][R.index(iNode)] = starS[key]

    flag = True  ##如果是度均值，则为true
    inferredE = []
    number = len(R) + 1  ##编号
    V = copy.copy(R)  ##所有的节点
    hatV = copy.copy(R)  ##
    hatS = np.zeros((len(R), len(R)))
    for iNode in range(len(R)):
        for jNode in range(len(R)):
            if iNode != jNode:
                X = copy.copy(S[iNode][jNode])
                if len(X) != 1:
                    X = list(X)
                    X.extend(S[jNode][iNode])
                else:
                    flag = False
                    X = [X]
                    X.append(S[jNode][iNode])
                hatS[iNode][jNode] = norm.fit(X)[0]
    dotS = copy.copy(hatS)
    while len(hatV) > 1:
        for inode in V:
            for jnode in V:
                if inode in hatV and jnode in hatV:
                    continue
                else:
                    hatS[V.index(inode)][V.index(jnode)] = hatS[V.index(jnode)][V.index(inode)] = 0
        indexs = np.where(np.max(hatS) == hatS)[0]
        iIndex = indexs[0]
        jIndex = indexs[1]
        V.append(number)
        hatV.remove(V[iIndex])
        hatV.remove(V[jIndex])
        inferredE.append((number, V[iIndex]))
        inferredE.append((number, V[jIndex]))

        tempS = np.zeros((len(hatS[0]), len(hatS[0])))
        for iNode in range(len(hatS[0])):
            for jNode in range(len(hatS[0])):
                tempS[iNode][jNode] = hatS[iNode][jNode]
        hatS = np.zeros((len(hatS[0]) + 1, len(hatS[0]) + 1))
        for iNode in range(len(tempS[0])):
            for jNode in range(len(tempS[0])):
                hatS[iNode][jNode] = tempS[iNode][jNode]
        for node in hatV:
            if node not in R:
                descendants1 = getDescendants(inferredE, node)
            else:
                descendants1 = [node]
            for lNode in descendants1:
                descendants2 = getDescendants(inferredE, number)
                X = []
                if flag:
                    for rNode in descendants2:
                        X.extend(S[V.index(lNode)][V.index(rNode)])
                        X.extend(S[V.index(rNode)][V.index(lNode)])
                else:
                    for rNode in descendants2:
                        X.append(S[V.index(lNode)][V.index(rNode)])
                        X.append(S[V.index(rNode)][V.index(lNode)])
            hatS[V.index(number)][V.index(node)] = hatS[V.index(node)][V.index(number)] = norm.fit(X)[0]
        hatV.append(number)
        number += 1
    inferredE.append((0, hatV[0]))
    inferredE = numberTopo(inferredE, R)
    return inferredE, dotS
def prune(inferredBE,S,R,e):
    '''
    比较通用的剪枝
    :param inferredBE:
    :param S:
    :param R:
    :return:
    '''
    newE = []
    # TreePlot(EtoVTree(inferredBE))
    threshold = e
    U = [getChildren(inferredBE,0)[0]]
    newE.append((0,U[0]))
    while U:
        parent = U[0]
        del U[0]
        children = getChildren(inferredBE,parent)
        for child in children:
            if child not in R:
                if iscut(parent,child,inferredBE,S,R,threshold):##做删除操作
                    cut(parent,child,inferredBE,newE,U,R,S,threshold)
                else:
                    newE.append((parent, child))
                    U.append(child)
            else:
                newE.append((parent,child))
    inferredE = numberTopo(newE,R)
    return inferredE

def cut(parent,child,inferredBE,newE,U,R,S,threshold):
    grandchildren = getChildren(inferredBE, child)
    for node in grandchildren:
        if node not in R:
            if iscut(parent,node,inferredBE,S,R,threshold):
                cut(parent,node,inferredBE,newE,U,R,S,threshold)
            else:
                newE.append((parent,node))
                U.append(node)
        else:
            newE.append((parent, node))

def iscut(parent, child, inferredBE, S, R, threshold):
    '''
    判断当前的边需不需要被砍掉
    :param parent:
    :param child:
    :param inferredBE:
    :param S:
    :param threshold:
    :return:
    '''
    ##得到两个内节点公共路径长度差
    descendants1 = getDescendants(inferredBE, parent)
    datalist1 = []
    descendants2 = getDescendants(inferredBE, child)
    datalist2 = []
    for child1 in descendants1:
        for child2 in descendants1:
            if child1 != child2:
                if parent == getAncestor(inferredBE, child1, child2):
                    datalist1.append(S[R.index(child1)][R.index(child2)])
    for child1 in descendants2:
        for child2 in descendants2:
            if child1 != child2:
                if child == getAncestor(inferredBE, child1, child2):
                    datalist2.append(S[R.index(child1)][R.index(child2)])
    if len(datalist1) == 0:
        print(parent, child, inferredBE)
        sys.exit(0)
    mean1 = np.mean(datalist1)
    mean2 = np.mean(datalist2)
    if np.abs(mean1 - mean2) <= threshold:  ##做删除操作
        return True
    else:
        return False

def plotResult(plot_data=[],FILENAME={},file=False, plot_alg="ALL"):
    '''
    画出结果,可以选择从文件中，或者从变量中获取数据，默认从变量中，默认画出所有曲线
    :param plot_data: 从变量中获取数据时，变量保存在这里面
    :param FILENAME: 从文件中获取数据时，文件路径保存在这个字典中
    :param file: 否是从文件中读取数据
    :param plot_alg: 要画的图中包含的算法
    :return:
    '''
    ALG = []
    if plot_alg == "ALL":
        ALG = ["ALT", "RNJ", "T-test", "HTE"]
        # ALG = ["ALT","RNJ"]
        # ALG = ["HTE"]
        ALG_line = ["x-",".-","o-","v-"]
    if plot_alg != "ALL":
        for alg in plot_alg.split("|"):
            ALG.append(alg)
    if file:
        ##获取文件长度
        lenght = FILENAME["len"]
        PATHNUM = [i + 3 for i in range(int(lenght/100))]
        ##画精度图
        fig = plt.subplot()
        for alg in ALG:
            acc = np.loadtxt(FILENAME[alg]["acc"])
            ACC = []
            for i in range(int(lenght/100)):
                ACC.append(np.mean(acc[i*100:(i+1)*100]))
            plt.plot(PATHNUM,ACC,ALG_line[ALG.index(alg)],label=alg)
        plt.xlabel("PathNum")
        plt.ylabel("PC")
        ALG_line[ALG.index(alg)]
        plt.legend()
        plt.title("")
        plt.show()
        plt.close()

        ##画编辑距离图
        fig = plt.subplot()
        for alg in ALG:
            ed = np.loadtxt(FILENAME[alg]["ed"])
            ED = []
            for i in range(int(lenght/100)):
                ED.append(np.mean(ed[i*100:(i+1)*100]))
            plt.plot(PATHNUM, ED,ALG_line[ALG.index(alg)], label=alg)
        plt.xlabel("PathNum")
        plt.ylabel("Tree edit distance")
        plt.xticks(PATHNUM)
        plt.legend()
        plt.title("")
        plt.show()
        plt.close()
    else:
        ##输出编辑距离的图
        fig = plt.subplot()
        for alg in ALG:
            edit_distance = plot_data[alg]["edit_distance"]
            PATHNUM = [i+5 for i in range(9)]
            ED = []
            for pathNum in PATHNUM:
                index = PATHNUM.index(pathNum)
                mean_ed = np.mean(edit_distance[index*100:(index+1)*100])
                ED.append(mean_ed)
            plt.plot(PATHNUM,ED,label=alg)
        plt.xlabel("pathNum")
        plt.ylabel("edit_distance")
        plt.title("")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()
        plt.close()

        fig = plt.subplot()
        for alg in ALG:
            correct = plot_data[alg]["correct"]
            PATHNUM = [i+5 for i in range(9)]
            ACC = []
            for pathNum in PATHNUM:
                index = PATHNUM.index(pathNum)
                mean_acc = np.mean(correct[index * 100:(index + 1) * 100])
                ACC.append(mean_acc)
            plt.plot(PATHNUM, ACC, label=alg)
        plt.xlabel("pathNum")
        plt.ylabel("accuracy")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title("")
        plt.show()
        plt.close()
def calMetrics(pktnum=200):
    '''
    计算度均值
    :param pktnum:
    :return:
    '''
    VTrees = getVTrees()
    n = len(VTrees)
    while len(VTrees) != 0:
        ##还有文件没处理完
        flag = False ##表示没有文件要处理
        f1 = " "
        f2 = " "
        for i in range(n):
            filename1 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/sourceTrace" + str(
                i) + ".tr"
            filename2 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/Metric" + str(i)
            filename3 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/sourceTrace" + str(
                i+1) + ".tr"
            flag1 = os.path.exists(filename1)
            flag2 = os.path.exists(filename2)
            flag3 = os.path.exists(filename3)
            if flag1 == True and flag2 == False and flag3 == True:

                print("正在处理sourceTrace" + str(i) + "...")
                f1 = filename1
                f2 = filename2
                flag = True
                serial_number = i
                break
            elif flag1 == True and flag2 == False and flag3 == False:
                # print("还未生成sourceTrace" + str(i) + "...")
                print("waiting for sourceTrace"+str(i+1)+"...")
                time.sleep(900)
                while os.path.exists(filename3) == False:
                    print("waiting for sourceTrace" + str(i+1) + "...")
                    time.sleep(900)
                print("正在处理sourceTrace" + str(i) + "...")
                f1 = filename1
                f2 = filename2
                flag = True
                break
            elif flag1 == False and flag2 == True:
                continue
        if flag:
            S = calmetric(f1,VTrees[i])
            saveS(S,f2)
            os.remove(f1)

def saveS(S,filename):
    if os.path.exists(filename):
        os.remove(filename)
    for key in S:
        open(filename,"a+").write(str(S[key]))
        open(filename, "a+").write("\n")
def calmetric(filename,VTree,pktnum=200):
    '''
    计算单个的sourceTrace文件
    保存数据的格式：
    {
        "1,2":{
            "packet_order":[[packet_id,  ,  ]
            ]
        }
    }
    :return:
    '''
    R = getLeafNodes(VTree)  ##保存叶子节点
    n = len(R)
    srcIP = "10.1."+str(n+1)+".1"
    destIPList = []
    for item in R:
        destIPList.append("10.1."+str(item)+".2")
    # S = [[ [] for j in range(n)] for i in range(n)]
    S = {}
    for i in R:
        for j in R:
            if j > i:
                key = str(i)+","+str(j)
                S[key] = []
    record ={}
    sum = (n-1+1)/2*(n-1)*pktnum
    cnt = 0
    for i in R:
        for j in R:
            if j > i:
                key = str(i)+","+str(j)
                record[key]={}
                record[key]["packet_order"] = []
                record[key]["num"] = 0
    sandwich = []
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            # print(line)
            if line:
                oc = re.split(r"\)|\(| ",line)
                action = oc[0]
                time = float(oc[1])
                namespace = oc[2]
                currentNode = int(namespace.split("/")[2])
                packet_id = int(oc[23])
                src_ip = oc[35]
                dest_ip = oc[37]
                src_port = oc[43]
                dest_port = oc[45]
                size = int(oc[49].split("=")[1])
                if currentNode==0 and src_ip==srcIP and dest_ip in destIPList and action=="+":
                    dest_node = int(dest_ip.split('.')[2])
                    ##记录根节点流出的三明治包流
                    sandwich_len = len(sandwich)
                    if sandwich_len >= 3:
                        print("sandwich length greater than or equal to 3!!")
                    if sandwich_len == 0:
                        #正在记录三明治包的第一个报文，验证
                        if size == 50:
                            sandwich.append([packet_id,dest_node])
                        else:
                            print("first packet in sandwich packet lost in 0 node!!")
                    if sandwich_len == 1:
                        ##正在记录三明治包的第二报文
                        if size == 1400:
                            sandwich.append([packet_id,dest_node])
                        else:
                            print("second packet in sandwich packet lost in 0 node!!")
                    if sandwich_len == 2:
                        ##正在记录三明治包中的第三个报文
                        if size == 50 and packet_id == sandwich[0][0]+1 and dest_node == sandwich[0][1]:
                            sandwich.append([packet_id,dest_node])
                            key = str(sandwich[0][1])+","+str(sandwich[1][1])
                            record[key]["packet_order"].append([sandwich[0][0],sandwich[1][0],packet_id])
                            record[key]["num"] = record[key]["num"]+1
                            sandwich = []
                        else:
                            print("third packet in sandwich packet lost in 0 node!!")
                if currentNode in R and src_ip==srcIP and dest_ip in destIPList and action=="r":
                    toRemove = []
                    dest_node = int(dest_ip.split('.')[2])
                    ##当前节点为接受节点
                    if size == 50:
                        ##此时接受小包,找到这个小包的位置，用时间去代替
                        flag1 = True
                        for key in record:
                            ##根据发送的设置，小包只会发送到节点对中的第一个位置，如“1,2”，则小包发送到1
                            if int(key.split(",")[0]) == dest_node:
                                for item in record[key]["packet_order"]:
                                    if packet_id in item:
                                        index = item.index(packet_id)
                                    else:
                                        continue
                                    ##先进行最基本的错误处理,处理最有可能的情况
                                    if index == 2 and isinstance(item[0],int):
                                        ##此时此三明治包中第一个包丢失
                                        print("first packet in sandwich packet lost in transmision!!")
                                    if index == 1 and item[2] == packet_id:
                                        index = 2 ## index=1 表示1位置已经换成了time 且此时 第三位置id与当前id相同
                                        item[index] = time
                                        if isinstance(item[0], float) and isinstance(item[1], float) and isinstance(
                                                item[2], float):
                                            arrivalInterval = item[2] - item[0]
                                            # S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
                                            S[key].append(arrivalInterval)
                                            toRemove.append(key)
                                            toRemove.append(item)
                                        flag1 = False
                                        break
                                    if index == 0 or index == 2:
                                        item[index] = time
                                        if isinstance(item[0],float) and isinstance(item[1],float) and isinstance(item[2],float):
                                            arrivalInterval = item[2]-item[0]
                                            # S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
                                            S[key].append(arrivalInterval)
                                            toRemove.append(key)
                                            toRemove.append(item)
                                        flag1 = False
                                        break
                            if flag1 == False:
                                break
                    if size == 1400:
                        dest_node = int(dest_ip.split('.')[2])
                        ##此时接受第二个报文，找到位置时间取替换
                        flag2 = True
                        for key in record:
                            if int(key.split(",")[1]) == dest_node:
                                for item in record[key]["packet_order"]:
                                    if item[1] == packet_id:
                                        item[1] = time
                                        if isinstance(item[0], float) and isinstance(item[1], float) and isinstance(
                                                item[2], float):
                                            arrivalInterval = item[2] - item[0]
                                            # S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
                                            S[key].append(arrivalInterval)
                                            toRemove.append(key)
                                            toRemove.append(item)
                                            flag2 == False
                                            break
                                    else:
                                        continue
                            if flag2 == False:
                                break
                    if len(toRemove) == 2:
                        record[toRemove[0]]["packet_order"].remove(toRemove[1])
                        cnt = cnt + 1
                        if toRemove[0] == "4,8":
                            print(toRemove[0],toRemove[1],[toRemove[1][2]-toRemove[1][0]-0.02],cnt)
                        if cnt == sum:
                            print("all receive")
                            break
                        if cnt == sum-1:
                            pass
            else:
                break
    '''
    除输出外剩余的丢失情况在record中，num参数也可以进行操作来得知丢失情况 这里时间有限，没有写这部分，但是结果中考虑了丢失情况
    '''
    return S
def saveFile(filename,data):
    '''
    保存文件
    :param filename:
    :param data:
    :return:
    '''
    print("正在保存文件：",filename)
    np.savetxt(filename,data)

def evaluation(data_dir,flag = False):
    ##记录文件长度
    lenght = 800
    ##记录文件地址
    FILENAME = {
    }
    ALG = ["ALT", "RNJ", "T-test", "HTE"]
    # ALG = ["ALT", "RNJ"]
    # ALG = ["HTE"]
    for alg in ALG:
        FILENAME[alg]={}
        filename1 = data_dir+"/"+alg+"/inferredE_"+alg
        filename2 = data_dir+"/SourceE"
        filename3 = data_dir+"/"+alg+"/"+"ACC_"+alg
        filename4 = data_dir+"/"+alg+"/"+"ED_"+alg
        FILENAME[alg]["acc"] = filename3
        FILENAME[alg]["ed"] = filename4
        if flag:
            if os.path.exists(filename3):
                os.remove(filename3)
            if os.path.exists(filename4):
                os.remove(filename4)
            acc = []
            ed = []
            f1 = open(filename1,"r")
            f2 = open(filename2,"r")
            while True:
                line1 = f1.readline()
                line2 = f2.readline()
                if line1 and line2:
                    if line1 == line2:
                        acc.append(1)
                        ed.append(0)
                    else:
                        E1 = ast.literal_eval(line1)
                        E2 = ast.literal_eval(line2)
                        acc.append(0)
                        ed.append(calEDbyzss(E1,E2))
                else:
                    break
            if filename3 != "" and filename4 != "":
                lenght = len(acc)
                np.savetxt(filename3,acc)
                np.savetxt(filename4,ed)
    FILENAME["len"] = lenght
    return FILENAME

def doSim(data_dir,getMetric,flag=False):
    '''
    获取度量，执行算法函数，保存推断的E和编辑距离，以及精确度，画图
    :param data_dir:
    :return:
    '''
    if(flag):##推断
        # ALG = ["ALT", "RNJ", "T-test","HTE"]
        ALG = [ "RNJ","ALT"]
        # ALG = ["HTE"]
        # ALG = ["T-test","HTE"]
        for alg in ALG:
            filename = data_dir+"/"+alg+"/"+"inferredE_"+alg  ##清理一下
            if os.path.exists(filename):
                os.remove(filename)
        PATHNUM = [i + 3 for i in range(8)]
        # PATHNUM = [i + 5 for i in range(8)]
        VTrees = getVTrees(data_dir+"/Topo_4_3_10")
        # VTrees = getVTrees(data_dir + "/VTrees_5_5_13")
        sourceE = getSourceEs(data_dir+"/SourceE")
        serial_number = 0
        for VTree in VTrees:
            print(str(serial_number),sourceE[VTrees.index(VTree)])
            pathNum = PATHNUM[int(serial_number / 100)]
            filename = data_dir + "/" +str(pathNum)+ "/Metric" + str(serial_number)
            if os.path.exists(filename):
                for alg in ALG:
                    if alg == "RNJ":
                        ##处理文件
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R),2000,2000)
                        # e = 0.0003813334
                        e = 1.92e-08
                        inferredE = RNJ(R,S,e)
                        filename1 = data_dir+"/"+alg+"/inferredE_"+alg
                        open(filename1,"a+").write(str(inferredE)+"\n")
                        print("RNJ:",inferredE)
                    else:
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R))
                        if alg == "ALT":
                            # e = 0.0003813334
                            e = 1.92e-08
                            inferredBE,dotS = ALT(S,R)
                            inferredE = prune(inferredBE,dotS,R,e)
                            filename2 = data_dir + "/" +alg+ "/inferredE_" + alg
                            open(filename2, "a+").write(str(inferredE) + "\n")
                            print("ALT:", inferredE)
                        elif alg == "HTE":
                            transform(S)
                            inferredE = HTE(S,R)
                            filename3 = data_dir + "/" +alg+ "/inferredE_" + alg
                            open(filename3, "a+").write(str(inferredE) + "\n")
                            print("HTE:", inferredE)
                        elif alg == "T-test":
                            transform(S)
                            inferredBE = ALT(S,R)[0]
                            inferredE = TP(inferredBE,R,S)
                            filename3 = data_dir + "/" + alg+"/inferredE_" + alg
                            open(filename3, "a+").write(str(inferredE) + "\n")
                            print("T-test:", inferredE)
            else:
                ##文件不存在
                print("file not exists:",filename)
                sys.exit(0)

            serial_number = serial_number+1
    FILENAME = evaluation(data_dir,flag)
    plotResult(FILENAME=FILENAME,file=True)
def transform(S):
    for key in S:
        for i in range(len(S[key])):
            S[key][i] = S[key][i]*1000000
def genSourceEs(data_dir):
    '''
    生成sourceE文件
    :param filename:
    :return:
    '''
    FILENAME = data_dir+"/SourceE"
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
    f = open(FILENAME,"a+")
    filename = data_dir+"/VTrees_5_5_13"
    VTrees = getVTrees(filename)
    for VTree in VTrees:
        sourceE = numberTopo(VTreetoE(VTree), getLeafNodes(VTree))
        f.write(str(sourceE))
        f.write('\n')

def getMetric(filename,n,K=200,Norm=20):
    '''

    :param filename:
    :param n: 叶节点个数
    :param K:
    :param Norm:
    :return:
    '''
    SIJ = []
    f = open(filename,"r")
    while True:
        line = f.readline()
        if line:
            sij = ast.literal_eval(line)
            if len(sij) == K:
                if K == Norm:
                    ## 全部求均值
                    SIJ.append(np.mean(sij))
                else:
                    ##求度均值
                    temp = []
                    for i in range(int(K/Norm)):
                        temp.append(np.mean(sij[i*Norm:(i+1)*Norm]))
                    SIJ.append(temp)
            else:
                print("getMetic中发现存在丢包")
                if K == Norm:
                    ## 全部求均值
                    SIJ.append(np.mean(sij))
                else:
                    ##求度均值
                    temp = []
                    partition = int(K/Norm)
                    norm = int(len(sij)/partition)
                    for i in range(partition):
                        if i+1 == partition:
                            temp.append(np.mean(sij[i * norm:]))
                        else:
                            temp.append(np.mean(sij[i*norm:(i+1)*norm]))
                    SIJ.append(temp)

        else:
            break
    if K == Norm:
        S = np.zeros((n, n))
        index = 0
        for i in range(n):
            for j in range(n):
                if j>i:
                    S[i][j] = S[j][i] = (SIJ[index]-0.02)
                    index = index+1
    else:
        S = {}
        index = 0
        for i in range(n):
            for j in range(n):
                if j>i:
                    key = "S"+str(i+1)+","+str(j+1)
                    if index >= len(SIJ):
                        print("error")
                    for k in range(len(SIJ[index])):
                        SIJ[index][k] = SIJ[index][k]-0.02
                    S[key] = SIJ[index]
                    index = index+1
    return S
def checkFile():
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3"
    for i in range(900):
        filename = data_dir+"/Metric"+str(i)
        if os.path.exists(filename):
            print("正在检查:  ",filename)
            f = open(filename,"r")
            while True:
                line = f.readline()
                if line:
                    if len(ast.literal_eval(line)) != 200:
                        print("缺少")
                else:
                    break

def getSourceEs(filename):
    sourceE = []
    f = open(filename, "r")
    while True:
        line = f.readline()
        if line:
            sourceE.append(ast.literal_eval(line))
        else:
            break
    return sourceE
# def test(data_dir):
#     VTrees = getVTrees(data_dir+"/VTrees_5_5_13")
#     serial_number = 0
#     PATHNUM = [i + 5 for i in range(9)]
#     for VTree in VTrees:
#         pathNum = PATHNUM[int(serial_number / 100)]
#         filename = data_dir+"/"+str(pathNum)+"/"+"Metric"+str(serial_number)
#         if os.path.exists(filename):
#             R = getLeafNodes(VTree)
#             S = getMetric(filename,len(R))
#             inferredE = HTE(S,R)
#             serial_number = serial_number+1
# def test2(filename="/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/test_4.tr"):
#     record = []
#     lines = open(filename,"r").readlines()
#     for line in lines:
#         oc = re.split(r"\)|\(| ", line)
#         action = oc[0]
#         time = float(oc[1])
#         namespace = oc[2]
#         currentNode = int(namespace.split("/")[2])
#         packet_id = int(oc[23])
#         src_ip = oc[35]
#         dest_ip = oc[37]
#         src_port = oc[43]
#         dest_port = oc[45]
#         # size = int(oc[49].split("=")[1])
#         if currentNode == 2 and action == "r" and src_ip == "10.0.1.1" and dest_ip == "10.1.2.2":
#             print("packet id:",packet_id,"receive time:",time)
#             if packet_id%2 != 0 and record:
#                 print("时间:",time-record[0]-0.02)
#                 del record[0]
#             else:
#                 record.append(time)
#
#     print("done")

if __name__ == "__main__":
    # calMetrics()
    # doSim("/media/zongwangz/RealPAN-13438811621/myUbuntu/data/alg",getMetric,False)
    # test("/media/zongwangz/RealPAN-13438811621/myUbuntu/data/alg")
    # S = calmetric("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/sourceTrace0.tr",[4, 5, 5, 0, 4],200)
    # saveS(S,"temp")
    # saveS(S,"/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/Metric566_1.tr")
    # S = {"S1,2":stats.gennorm.rvs(0.36267604204732506,0.002922659999999873,1.0521293968907097e-06,10),
    #      "S1,3":stats.gennorm.rvs(0.36267604204732506,0.002922659999999873,1.0521293968907097e-06,10),
    #     "S2,3":stats.gennorm.rvs(0.11041748842287355,0.005802670659316566,3.508930776095499e-14,10)}
    # S = {"S1,2": stats.norm.rvs(0.002922659999999873, 1.0521293968907097e-06, 10),
    #      "S1,3": stats.norm.rvs(0.002922659999999873, 1.0521293968907097e-06, 10),
    #      "S2,3": stats.norm.rvs(0.005802670659316566, 3.508930776095499e-14, 10)}
    # print(HTE(S,[1,2,3]))
    # pass
    # test2()
    # S = getMetric("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/light_load/9/Metric661",9)
    # transform(S)
    # R = [1,2,3,4,5,6,7,8,9]
    # inferredBE = ALT(S,R)[0]
    # inferredE = TP(inferredBE,R,S)
    pass
