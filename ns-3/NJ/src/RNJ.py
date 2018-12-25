import numpy as np

from collections import deque
from tool import *
import copy
import json
from matplotlib import pyplot as plt
class RNJ:
    @staticmethod
    def RNJ(data):
        R = copy.copy(data['R'])
        S = data['S']
        e = data['e']
        dotR = R.copy()
        V = [0]
        E = []
        n = len(dotR)
        number = n+1
        pathDistance = []
        for i in range(n):
            pathDistance.append(S[i][i])
            S[i][i] = 0
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
                if S[iIndex][jIndex]-S[iIndex][kIndex]<e:
                    brother.append(kNode)
                    V.append(kNode)
                    E.append((number,kNode))
            for node in brother:
                dotR.remove(node)
            n = len(R)
            tempS = np.zeros((n, n))
            for i in range(n-1):
                for j in range(n - 1):
                    tempS[i][j] = S[i][j]
            S = tempS
            for node in dotR:
                index = R.index(node)
                S[n-1][index] = S[index][n-1] = S[iIndex][index]
            dotR.append(number)
            number = number+1
        E.append((0,dotR[0]))
        inferredE = numberTopo(E,data['R'])
        data['inferredE'] = inferredE
        return E

    @staticmethod
    def genData(outDegree = 5,pathNum = 12,scale = 1.0,probesNum = 500):
        ## gen VTree
        VTree = GenTree(outDegree,pathNum)
        E = numberTopoByVTree(VTree)
        VTree = EtoVTree(E)
        R = getLeafNodes(VTree)
        n = len(R)
        ## gen linkDelay
        linkDelay = np.random.exponential(scale,(len(VTree),probesNum))
        linkdistance = []
        for i in range(len(VTree)):
            linkdistance.append(Variance_way1(linkDelay[i]))
        e = min(linkdistance)/2
        RM = getRM(R,VTree)
        pathDelay = np.dot(RM,linkDelay)
        ## gen sharePathDistance
        S = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                S[i][j] = Covariance_way2(pathDelay[i],pathDelay[j])
        data = {
            'VTree':VTree,
            'E':E,
            'R':R,
            'S':S,
            'e':e
        }
        return data



    @staticmethod
    def doSim():
        PC = []     ## 精度
        ED = []     ## 编辑距离
        PN = [_ for _ in range(5, 13)]
        for pathNum in PN:
            cnt = 0
            edit_distance = []
            listObj = []
            for i in range(100):
                data = RNJ.genData(outDegree=5, pathNum=pathNum, scale=1.0, probesNum=500)
                RNJ.RNJ(data)
                E = data['E']
                inferredE = data['inferredE']
                ed = calEDbyzss(E,inferredE)
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
            # with open('/home/zongwangz/文档/Projects/NJ/data/NJ_' + str(pathNum), 'w') as f:
            #     f.write(jsonObj)
            #     f.write('\n')
            #     f.write("mean edit distance:")
            #     f.write(str(np.mean(edit_distance)))
            #     f.write('\n')
            #     f.write("PC:")
            #     f.write(str(cnt/100))
            #     f.write('\n')
            ED.append(np.mean(edit_distance))
            print(np.mean(edit_distance))
            PC.append(cnt / 100)
            print(cnt / 100)

        fig1 = plt.subplot()
        plt.xlabel('pathNum')
        plt.ylabel('edit distance')
        plt.plot(PN, ED, 'o-', label='NJ')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()

        fig2 = plt.subplot()
        plt.xlabel('pathNum')
        plt.ylabel('PC')
        plt.plot(PN, PC, 'o-', label='NJ')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()


if __name__ == '__main__':

    # RNJ.getResults()
    RNJ.doSim()
    # data = RNJ.genData(outDegree=5, pathNum=12, scale=1.0, probesNum=40000)
    # print(data['S'])
