from tool import *
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
import os
import copy
import json
import math
from sim_tool import *
parameter = {
    "probe_way":"sandwich",
    "data_way":"normal"
}
class ALT:
    @staticmethod
    def test():
        data = norm.rvs(loc=1, scale=0.1, size=2)
        print(data)
        fit_loc, fit_beta = norm.fit(data)
        print(fit_loc, fit_beta)

    @staticmethod
    def getThreshold(S):
        # ToChoose = []
        # n = len(S[0])
        # for i in range(n):
        #     j = i+1
        #     while j < n:
        #         k = j+1
        #         while k < n:
        #             print(str(i+1)+str(j+1)+"-"+str(i+1)+str(k+1))
        #             ToChoose.append(np.abs(S[i][j]-S[i][k]))
        #             k += 1
        #         j += 1
        return 0.5


    @staticmethod
    def prune(inferredBE,S,R):
        '''
        比较通用的剪枝
        :param inferredBE:
        :param S:
        :param R:
        :return:
        '''
        newE = []
        # TreePlot(EtoVTree(inferredBE))
        threshold = ALT.getThreshold(S)
        U = [getChildren(inferredBE,0)[0]]
        newE.append((0,U[0]))
        while U:
            parent = U[0]
            del U[0]
            children = getChildren(inferredBE,parent)
            for child in children:
                if child not in R:
                    if ALT.iscut(parent,child,inferredBE,S,R,threshold):##做删除操作
                        ALT.cut(parent,child,inferredBE,newE,U,R,S,threshold)
                    else:
                        newE.append((parent, child))
                        U.append(child)
                else:
                    newE.append((parent,child))
        inferredE = numberTopo(newE,R)
        return inferredE

    @staticmethod
    def cut(parent,child,inferredBE,newE,U,R,S,threshold):
        grandchildren = getChildren(inferredBE, child)
        for node in grandchildren:
            if node not in R:
                if ALT.iscut(parent,node,inferredBE,S,R,threshold):
                    ALT.cut(parent,node,inferredBE,newE,U,R,S,threshold)
                else:
                    newE.append((parent,node))
                    U.append(node)
            else:
                newE.append((parent, node))


    @staticmethod
    def iscut(parent,child,inferredBE,S,R,threshold):
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
            print(parent,child,inferredBE)
            sys.exit(0)
        mean1 = np.mean(datalist1)
        mean2 = np.mean(datalist2)
        if np.abs(mean1 - mean2) <= threshold:  ##做删除操作
            return True
        else:
            return False

    @staticmethod
    def ALT(S,R):
        flag = True ##如果是度均值，则为true
        inferredE = []
        number = len(R)+1   ##编号
        V = copy.copy(R) ##所有的节点
        hatV = copy.copy(R)  ##
        hatS = np.zeros((len(R),len(R)))
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
            inferredE.append((number,V[iIndex]))
            inferredE.append((number,V[jIndex]))

            tempS = np.zeros((len(hatS[0]),len(hatS[0])))
            for iNode in range(len(hatS[0])):
                for jNode in range(len(hatS[0])):
                    tempS[iNode][jNode] = hatS[iNode][jNode]
            hatS = np.zeros((len(hatS[0])+1,len(hatS[0])+1))
            for iNode in range(len(tempS[0])):
                for jNode in range(len(tempS[0])):
                    hatS[iNode][jNode] = tempS[iNode][jNode]
            for node in hatV:
                if node not in R:
                    descendants1 = getDescendants(inferredE,node)
                else:
                    descendants1 = [node]
                for lNode in descendants1:
                    descendants2 = getDescendants(inferredE, number)
                    X =[]
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
        inferredE.append((0,hatV[0]))
        inferredE = numberTopo(inferredE,R)
        return inferredE,dotS
    @staticmethod
    def genData(VTree):
        # TreePlot(VTree)
        E = VTreetoE(VTree)
        R = getLeafNodes(VTree)
        S = [ [0 for _ in range(len(R))] for _ in range(len(R))]
        for iNode in range(len(R)):
            for jNode in range(len(R)):
                if iNode != jNode:
                    S[iNode][jNode] = ALT.getSij(R[iNode],R[jNode],E,probe_way=parameter["probe_way"],data_way=parameter['data_way'])
        return S

    @staticmethod
    def getSij(iNode,jNode,E,probe_way="sandwich",data_way=parameter['data_way']):
        '''
        根据way的生成方式生成数据,获取度量
        :param iNode:
        :param jNode:
        :param E:
        :param way:
        :return:
        '''
        if probe_way == "sandwich" and data_way == "normal":
            length = getSharedPathLenbyNodes(E,iNode,jNode)
            linkInterval = np.random.normal(1,np.random.uniform(0,0.2),(length,200))
            pathInterval = np.sum(linkInterval,0)
            return pathInterval


    @staticmethod
    def doSim(file=True):
        SourceE = []  ##储存原始的E
        InferredE = []  ##储存推断的E
        edit_distance = [] ##储存所有的编辑距离
        correct = []   ##储存推断正确与否
        VTrees = getVTrees()
        for VTree in VTrees:
            R = getLeafNodes(VTree)
            sourceE = numberTopo(VTreetoE(VTree),R)
            SourceE.append(sourceE)
            S = ALT.genData(VTree)
            inferredBE, dotS = ALT.ALT(copy.copy(S), copy.copy(R))
            inferredE = ALT.prune(copy.copy(inferredBE),copy.copy(dotS),copy.copy(R))
            InferredE.append(inferredE)
            if sourceE != inferredE:
                print(sourceE,inferredE,VTree)
                sys.exit(0)
        for i in range(len(SourceE)):
            ed = calEDbyzss(SourceE[i], InferredE[i])
            edit_distance.append(ed)
            if ed == 0:
                correct.append(1)
            else:
                correct.append(0)

        if file: ##存储中间数据
            filename1 = "/home/zongwangz/PycharmProjects/data/ALT/sourceE"
            filename2 = "/home/zongwangz/PycharmProjects/data/ALT/inferredE"
            filename3 = "/home/zongwangz/PycharmProjects/data/ALT/edit_distance"
            filename4 = "/home/zongwangz/PycharmProjects/data/ALT/correct"
            np.savetxt(SourceE, filename1)
            np.savetxt(InferredE, filename2)
            np.savetxt(edit_distance, filename3)
            np.savetxt(correct, filename4)
        plot_data = {
            "ALT":{
                "sourceE": SourceE,
                "inferredE":InferredE,
                "edit_distance":edit_distance,
                "correct":correct
            }
        }
        plotResult(plot_data, plot_alg="ALT")


if __name__ == "__main__":
    ALT.doSim(file=False)
    # VTree = [7, 8, 8, 7, 6, 0, 6, 7]
    # R = getLeafNodes(VTree)
    # sourceE = numberTopo(VTreetoE(VTree), R)
    # S = ALT.genData(VTree)
    # inferredBE, dotS = ALT.ALT(copy.copy(S), copy.copy(R))
    # inferredE = ALT.prune(copy.copy(inferredBE), copy.copy(dotS), copy.copy(R))
    # print(sourceE,inferredE,VTree)