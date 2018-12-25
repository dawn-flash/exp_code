import numpy as np
from tool import *
from scipy import stats, spatial, cluster
from matplotlib import pyplot as plt
import copy
import json
class HTE:

    @staticmethod
    def genData3(outDegree=5, pathNum=12, scale=1.0, probesNum=200):
        # 随机生成topo
        E = numberTopoByVTree(GenTree(outDegree, pathNum))
        VTree = EtoVTree(E)
        # TreePlot(VTree)
        # VTree = [0,1,1,2,2,2,3,3,3,4,4,5,5,6,6,9,9]
        # E = numberTopoByVTree(VTree,0)
        # VTree = EtoVTree(E)

        linkDelay = gen_linkDelay(VTree, scale=scale, probesNum=probesNum)
        R = getLeafNodes(VTree)
        RM = getRM(R, VTree)
        pathDelay = np.dot(RM, linkDelay)
        S = np.zeros((len(R), len(R)))
        for i in range(len(R)):
            for j in range(len(R)):
                S[i][j] = Covariance_way2(pathDelay[i], pathDelay[j])
        dist = np.zeros((len(R), len(R)))
        for i in R:
            for j in R:
                if j > i:
                    dist[i-1][j-1] = 1/S[i-1][j-1]*10

        data = {
            'E': E,
            'VTree': VTree,
            'linkDelay': linkDelay,
            'R': R,
            'RM': RM,
            'S': S,
            'dist': dist
        }
        return data


    @staticmethod
    def HTE(data):
        '''
        通用部分，输入是点之间的距离，输出是dendrogram树状图
        :param data:
        :return:
        '''
        z = cluster.hierarchy.linkage(data['dist'], data['way'])
        data['z'] = z

    @staticmethod
    def toBTree(data):
        z = data['z']
        R = data['R']
        BE = []
        Z = []
        newN = len(R) + 1
        for item in data['z']:
            ## Z 的形式是：第一个cluster，第二个cluster，新形成的cluster，两个cluster的距离
            Z.append([int(item[0]) + 1, int(item[1]) + 1, newN])
            BE.append((newN, int(item[0]) + 1))
            BE.append((newN, int(item[1]) + 1))
            newN = newN + 1
        Z = np.array(Z)
        data['Z'] = Z
        BE.append((0, newN - 1))
        numberBE = numberTopo(BE, R)
        BVTree = EtoVTree(numberBE)
        # TreePlot(BVTree)
        data['BE'] = numberBE

    @staticmethod
    def merge3(data):
        threshold = HTE.get_t(data)
        data['threshold'] = threshold
        n = len(data['R']) + 1
        E = data['BE']
        newE = []
        root = 0
        U = [root]
        while len(U) != 0:
            parent = U[0]
            del U[0]
            children = getChildren(E, parent)
            if len(children) == 0:
                continue
            U.extend(children)
            if parent == root:
                newE.append((parent, children[0]))
                continue
            for child in children:
                if child < n:
                    U.remove(child)
                    newE.append((parent, child))
                    continue
                if HTE.T_F(data, E, parent, child, threshold):
                    ##不剪去
                    newE.append((parent, child))
                else:

                    HTE.cut(U, child, E, data, parent, threshold, newE)

        numberE = numberTopo(newE, data['R'])
        numberVTree = EtoVTree(numberE)
        # TreePlot(numberVTree)
        data['inferredE'] = numberE

    @staticmethod
    def cut(U, child, E, data, parent, threshold,newE):
        U.remove(child)
        post_children = getChildren(E, child)
        U.extend(post_children)
        for post_child in post_children:
            if post_child < len(data['R'])+1 or HTE.T_F(data, E, parent, post_child, threshold):
                newE.append((parent, post_child))
            else:
                HTE.cut(U,post_child,E,data,parent,threshold,newE)

    @staticmethod
    def T_F(data,E,parent,child, threshold):
        if child < len(data['R']):
            return True
        p_len = HTE.getSharePathDelaybyParent(data, E, parent)
        c_len = HTE.getSharePathDelaybyParent(data, E, child)
        if  c_len-p_len < threshold:
            #当前边应当剪去
            return False
        else:
            return True

    @staticmethod
    def getSharePathDelaybyParent(data, E, parent):
        S = data['S']
        spd = []
        children = getDescendants(E, parent)
        if len(children) == 0:
            return
        children.sort()
        for child1 in children:
            for child2 in children:
                if child2 > child1 and getAncestor(E, child2, child1) == parent:
                    spd.append(S[int(child1)-1][int(child2)-1])
        return np.mean(spd)

    @staticmethod
    def get_t(data):
        linkLen = []
        for delay in data['linkDelay']:
            linkLen.append(Variance_way1(delay))
        return 0.5*np.min(linkLen)

    @staticmethod
    def test3(outDegree = 5,pathNum = 12,probesNum=200,way='average'):
        data = HTE.genData3(outDegree=outDegree, pathNum=pathNum, probesNum=probesNum)
        data['way'] = way
        HTE.HTE(data)
        ## 构建二叉树
        HTE.toBTree(data)
        HTE.merge3(data)
        E = data['E']
        inferredE = data['inferredE']
        # if E == inferredE:
        #     return True
        # else:
        #     return False
        return data




    '''
    由于输入理解错误 现在先把输入修改为 时延协方差的倒数  把输出的dendrogram图  改为VTree  输出图
    '''
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
                data = HTE.test3(outDegree=5, pathNum=pathNum, probesNum=200)
                E = data['E']
                inferredE = data['inferredE']
                ed = calEDbyzss(E, inferredE)
                edit_distance.append(ed)
                if ed == 0:
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
            # with open('/home/zongwangz/文档/Projects/HTE/data/HTE0/HTE_' + str(pathNum), 'w') as f:
            #     f.write(jsonObj)
            #     f.write('\n')
            #     f.write("mean edit distance:")
            #     f.write(str(np.mean(edit_distance)))
            #     f.write('\n')
            #     f.write("PC:")
            #     f.write(str(cnt / 100))
            #     f.write('\n')
            ED.append(np.mean(edit_distance))
            print(np.mean(edit_distance))
            PC.append(cnt / 100)
            print(cnt / 100)

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
if __name__ == "__main__":
    HTE.do_sim()
    # data = HTE.test3(outDegree=3, pathNum=12, probesNum=1000)
    # E = data['E']
    # inferredE = data['inferredE']
    # if E == inferredE:
    #     print("True")