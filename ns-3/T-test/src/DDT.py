'''
The Deterministic Delay-Variance Tree algorithm DDT=TP(e)+DBDT
Deterministic Binary Delay-Variance Tree Classification Algorithm(DBDT)
Tree pruning algorithm(TP(e)).
'''
import numpy as np
import os
from tool import *
from scipy.stats import t as T_d
from scipy.stats import ttest_ind
from scipy.stats import levene
from scipy.stats import bartlett
import copy
import json
import matplotlib.pyplot as plt
import math
class DDT:
    @staticmethod
    def DBDT(R, S):
        '''
        输入叶节点集合R和共享路径长度S
        :param R:
        :param S:利用协方差求出来的共享路径长度
        :return: 返回边结点集合，链路集合，链路延迟集合
        test_data:
        D=[3,4,6,7,8]
        VTree = [0,1,1,2,2,2,5,5]
        S=[[0,1,1,1,1],[1,0,2,2,2],[1,2,0,2,2],[1,2,2,0,3],[1,2,2,3,0]]
        '''
        pathLen = []
        for i in range(len(R)):
            pathLen.append(S[i][i])
            S[i][i] = 0
        r = {}
        sizeR = len(R)
        number = len(R)+1
        S = np.array(S)
        dotR = R.copy()
        dotV = dotR.copy()
        L = []
        while len(dotR) > 1:
            index = np.where(S == np.max(S))
            u = index[0][0]+1
            v = index[1][0]+1
            U = number
            number = number + 1
            dotV.append(U)
            if u in dotR:
                dotR.remove(u)
            if v in dotR:
                dotR.remove(v)
            dotR.append(U)
            pathLen.append(S[u-1][v-1])
            tempS = np.zeros((len(dotV),len(dotV)))
            for i in range(len(S[0])):
                for j in range(len(S[0])):
                    tempS[i][j] = S[i][j]
            S = tempS
            for k in dotR:
                S[k-1][U-1] = S[U-1][k-1] = S[u-1][k-1]
            for i in range(len(S[0])):
                S[u-1][i] = S[i][u-1] = 0
                S[v-1][i] = S[i][v-1] = 0
            L.append((U, u))
            L.append((U, v))
            r[str(u)] = pathLen[u-1] - pathLen[U-1]
            r[str(v)] = pathLen[v-1] - pathLen[U-1]
        dotV.append(0)
        L.append((0, dotR[0]))
        r[str(dotR[0])] = 1
        return dotV, L, r

    @staticmethod
    def TP(V, L, r,R, S,a = 0.005,way = "exponential",E=[],K = 200,Norm=20,data_way = 1):
        '''
        剪除链路时延小于e的链路
        （默认根节点只有一个孩子节点）
        :param V:
        :param L:
        :param r:
        :param e:
        :return: V L r
        '''
        assert isinstance(r,dict)
        dotV = []
        dotV.append(V[len(V)-1])
        U = getChildren(L, dotV[0])
        dotV.append(U[0])
        dotL = []
        dotL.append((dotV[0],dotV[1]))
        while len(U) != 0:
            j = U[0]
            U.remove(j)
            U.extend(getChildren(L, j))
            # if r[str(j)] <= e and j not in R:
            if DDT.isZeroLink(L,j,S,a,E,K,Norm,data_way) and j not in R:
                children = getChildren(L, j)
                parent = getParent(dotL, j)
                for k in children:
                    dotL.append((parent, k))
                    if (parent, j) in dotL:
                        dotL.remove((parent, j))
                        del r[str(j)]
            else:
                children = getChildren(L, j)
                for k in children:
                    dotL.append((j, k))
                    dotV.append(k)
        return dotV, dotL, r

    @staticmethod
    def isZeroLink(L, child, S,a = 0.005,E=[],K = 100,Norm=20,data_way=1):
        '''
        Given a significance level α,
        H0 will be rejected if t0 >tα,v,
        where tα,v is the upper α critical point of the t-distribution with v degrees of freedom, i.e., α =P(t0 >tα,v).
        :return:
        '''
        #1.首先得到Γv的集合
        parent = getParent(L,child)
        Tparent = []
        Tchild = []
        descendant1 = getDescendants(L,parent)
        descendant1.sort()
        for i in descendant1:
            for j in descendant1:
                if j > i:
                    if getAncestor(L,i,j) == parent:
                        Tparent.extend(DDT.getSij(S,i-1,j-1,E,K,Norm,data_way))  ##这里i-1和j-1是因为编号规则决定了，有更加通用的方法
        if len(Tparent) == 0:
            return False
        descendant2 = getDescendants(L, child)
        descendant2.sort()
        for i in descendant2:
            for j in descendant2:
                if j > i:
                    if getAncestor(L, i, j) == child:
                        Tchild.extend(DDT.getSij(S,i-1,j-1,E,K,Norm,data_way))
        if len(Tchild) == 0:
            return False
        #2.分别计算T-test的输入值
        Tparent = np.array(Tparent)
        Tchild = np.array(Tchild)
        # equal_var = bartlett(Tparent,Tchild)[1]

        # if equal_var > 0.05:
        #     # return DDT.ttest_way1(Tparent,Tchild)
        #     print("True")
        # else:
        #     # return DDT.ttest_way2(Tparent,Tchild)
        #     print("false")
        return DDT.ttest_way4(Tparent, Tchild,a)

    @staticmethod
    def ttest_way1(X1,X2,alpha = 0.005):
        result = ttest_ind(X1,X2,equal_var= True) ## two tailed p value
        if result[1] > 2*alpha:
            return True
        else:
            return False
    @staticmethod
    def ttest_way2(X1,X2,alpha = 0.005):
        result = ttest_ind(X1, X2, equal_var=False) ## two tailed p value
        if result[1] > 2*alpha:
            return True
        else:
            return False
    @staticmethod
    def ttest_way3(X1,X2,alpha = 0.005):
        hatX1 = np.mean(np.array(X1))
        hatX2 = np.mean(np.array(X2))
        squareX1 = Variance_way1(X1)
        squareX2 = Variance_way1(X2)
        n1 = len(X1)
        n2 = len(X2)
        squareS = 1/(n1+n2-2)*((n1-1)*squareX1+(n2-1)*squareX2)
        t = (hatX2-hatX1)/np.sqrt(squareS*(1/n1+1/n2))
        df = n1+n2-2
        interval = T_d.interval(1-2*alpha,df) ## two tails
        if t > interval[1]:
            return False
        else:
            return True
    @staticmethod
    def ttest_way4(X1,X2,alpha = 0.005):
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
    @staticmethod
    def ttest_way5(X1,X2,alpha = 0.005):
        hatX1 = np.mean(np.array(X1))
        hatX2 = np.mean(np.array(X2))
        squareX1 = Variance_way1(X1)
        squareX2 = Variance_way1(X2)
        n1 = len(X1)
        n2 = len(X2)
        t = (hatX2 - hatX1) / np.sqrt(squareX1 / n1 + squareX2 / n2)
        df = n1 + n2 - 2
        interval = T_d.interval(1 - 2 * alpha, df)  ## two tails
        if t > interval[1]:
            return False
        else:
            return True
    @staticmethod
    def ttest_way6(X1,X2,alpha = 0.005):
        hatX1 = np.mean(np.array(X1))
        hatX2 = np.mean(np.array(X2))
        squareX1 = Variance_way1(X1)
        squareX2 = Variance_way1(X2)
        n1 = len(X1)
        n2 = len(X2)
        t = (hatX2 - hatX1) / np.sqrt(squareX1 / n1 + squareX2 / n2)
        df1 = np.square(squareX1 / n1 + squareX2 / n2)
        df2 = np.square(squareX1 ) / (n1-1) + np.square(squareX2 ) / (n2-1)
        df = df1 / df2
        interval = T_d.interval(1 - 2 * alpha, df)  ## two tails
        if t > interval[1]:
            return False
        else:
            return True


    @staticmethod
    def test(outDegree = 3,pathNum = 5,K = 200,a = 0.005,Norm = 20):
        # 1.生成topo
        VTree = GenTree(outDegree, pathNum)
        E = numberTopoByVTree(VTree)
        VTree = EtoVTree(E)
        leafNodes = getLeafNodes(VTree)
        # 2.生成时延
        linkDelay = np.array(gen_linkDelay(VTree,scale=1))
        RM = getRM(leafNodes,VTree)
        pathDelay = np.dot(RM,linkDelay)
        # 3.计算S
        l = len(pathDelay)
        S = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                if j != i:
                    S[i][j] = Covariance_way2(pathDelay[i], pathDelay[j])
                else:
                    S[i][i] = Covariance_way2(pathDelay[i], pathDelay[i])
        V0, L0, r0 = DDT.DBDT(leafNodes, copy.deepcopy(copy.deepcopy(S)))
        V, L, r = DDT.TP(V0, L0, r0, leafNodes, DDT.gen_S(VTree,leafNodes,K), a,"exponential",E,K,Norm,data_way = 0)  ## for cov

        newE = numberTopo(L,leafNodes)
        ROOT = 0
        return E, ROOT , newE
    @staticmethod
    def gen_S(VTree,R,K = 200,Norm = 20,way = "exponential"):
        '''
        生成字典S
        :param VTree:
        :param K:
        :param Norm:
        :return:
        '''
        dictS = {}
        if way == "exponential":
            for k in range(int(K/Norm)):
                # 2.生成时延
                linkDelay = np.array(gen_linkDelay(VTree))
                pathDelay = np.array(calPathDelay(VTree, linkDelay))
                # 3.计算S
                l = len(pathDelay)
                S = np.zeros((l, l))
                for i in range(l):
                    for j in range(l):
                        if j != i:
                            S[i][j] = Covariance_way2(pathDelay[i], pathDelay[j])
                        else:
                            S[i][i] = Covariance_way2(pathDelay[i], pathDelay[i])
                dictS['S' + str(k)] = S
            setS = []
            for k in range(int(K/Norm)):
                setS.append(dictS['S' + str(k)])
            setS = np.array(setS)
        elif way == "normal":
            pass
        return setS

    @staticmethod
    def getSij(S,iNode,jNode,E={},K=100,Norm = 20,data_way=1):
        sij = []
        if data_way == 0:
            k = len(S)
            for i in range(k):
                sij.append(S[i][iNode][jNode])
        else:
            iNode = iNode+1
            jNode = jNode+1
            length = getSharedPathLenbyNodes(E,iNode,jNode)
            temp = []
            for k in range(K):
                temp.append(np.random.normal(1.0+np.random.uniform(-0.2,0.2),0.1+np.random.uniform(0,0.1),length).sum())
                if (k+1)%Norm == 0:
                    sij.append(np.mean(np.array(temp)))
                    temp = []
        return sij


    @staticmethod
    def doSimWork():
        PN = [_ for _ in range(5,13)]
        outDegree = 5
        for pathNum in PN:
            listObj = []
            for count in range(100):
                sourceE,root, E = DDT.test(outDegree=outDegree,pathNum=pathNum,K=200)
                t1,t2 = toBracketString(sourceE,E)
                dictObj = {
                    "testID": count,
                    "t1": t1,
                    "t2": t2,
                    "d": 0
                }
                listObj.append(dictObj)
            jsonObj = json.dumps(listObj)
            with open('/home/zongwangz/文档/Projects/T-test/data/T-test'+str(pathNum), 'w') as f:
                f.write(jsonObj)

    @staticmethod
    def do_sim():
        PC = []  ## 精度
        ED = []  ## 编辑距离
        PN = [_ for _ in range(5, 13)]
        for pathNum in PN:
            cnt = 0
            edit_distance = []
            listObj = []
            for i in range(100):
                sourceE, root, E = DDT.test(outDegree=5, pathNum=pathNum, K=200)
                ed = calEDbyzss(sourceE, E)
                edit_distance.append(ed)
                if ed == 0:
                    cnt = cnt + 1
                t1, t2 = toBracketString(sourceE, E)
                dictObj = {
                    "testID": i,
                    "t1": t1,
                    "t2": t2,
                    "d": ed
                }
                listObj.append(dictObj)
            jsonObj = json.dumps(listObj)
            with open('/home/zongwangz/文档/Projects/T-test/data/T-test' + str(pathNum), 'w') as f:
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
        plt.plot(PN, ED, 'o-', label='T-test')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

        fig2 = plt.subplot()
        plt.xlabel('pathNum')
        plt.ylabel('PC')
        plt.plot(PN, PC, 'o-', label='T-test')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

if __name__ == "__main__":

    # DDT.test(5,13)
    # DDT.doSimWork()
    # DDT.getResults()
    # DDT.getAVaried()
    # DDT.smallTest()
    DDT.do_sim()