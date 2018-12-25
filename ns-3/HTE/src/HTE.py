from tool import *
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import copy
import networkx as nx
import json
parameter = {
    'K':200,
    'Norm':20,
    'Nij':10,
    'probesNum':200,
}
class HTE:

    @staticmethod
    def genData(pathNum=8, outDegree=5, ):
        data = {}
        VTree0 = GenTree(outDegree=outDegree,pathNum=pathNum)
        # plot_Tree(copy.copy(VTree0),np.zeros(len(VTree0)))
        E = numberTopoByVTree(VTree0)
        VTree = EtoVTree(E)
        # VTree = [14,14,13,15,15,16,17,17,18,18,18,0,12,13,13,12,16,16]
        R = getLeafNodes(VTree)
        data['VTree'] = VTree
        data['E'] = E
        data['R'] = R
        S = HTE.genT(data)
        return data

    @staticmethod
    def genT(data,distribution='exponential'):
        '''
        使用200个背靠背包的协方差模拟一个度量，在一对叶节点上产生200个度量，每20个计算一次度均值，
        一对叶节点有10个度均值，背靠背包的时延由默认的指数分布生成，参数scale。
        :return:
        '''
        S = {}
        R = data['R']
        R.sort()
        for inode in R:
            for jnode in R:
                if jnode > inode:
                    key = 'S'+str(inode)+','+str(jnode)
                    S[key] = HTE.getTij(inode, jnode, data, distribution=distribution)
        data['S'] = S
        return S

    @staticmethod
    def getTij(inode, jnode, data, distribution='exponential'):
        '''
        选择两条路径，在两条路径的公共路径上生成相同的时延，在分支节点到目的节点上生成各自的时延
        :param inode:
        :param jnode:
        :param data:
        :param distribution:
        :return:
        '''
        K = 200
        Norm = 20
        E = data['E']
        shareLen = getSharedPathLenbyNodes(E, inode, jnode)
        branch_node = getAncestor(E, inode, jnode)
        branch_len1 = getLinkNumBetweenNodes(E, inode, branch_node)
        branch_len2 = getLinkNumBetweenNodes(E, jnode, branch_node)
        Tij = []
        tempij = []
        for i in range(K):
            temp1 = np.random.exponential(1.0, (shareLen, 200))
            share_PathDelay = np.sum(temp1, 0)
            temp2 = np.random.exponential(1.0, (branch_len1, 200))
            temp3 = np.random.exponential(1.0, (branch_len2, 200))
            branch_pathDelay1 = np.sum(temp2, 0)
            branch_pathDelay2 = np.sum(temp3, 0)
            pathDelay1 = np.sum(np.array([share_PathDelay,branch_pathDelay1]), 0)
            pathDelay2 = np.sum(np.array([share_PathDelay, branch_pathDelay2]), 0)
            gamaij = Covariance_way2(pathDelay1, pathDelay2)
            tempij.append(gamaij)
            if (i+1)%Norm == 0:
                Tij.append(np.mean(tempij))
                tempij = []
        return Tij

    @staticmethod
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
                    W = HTE.calW(Tij, label)
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
    @staticmethod
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

    @staticmethod
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
            K_dict['K'+str(i+1)] = HTE.precluster_Alg(dest_node, key_data, B_dict['B'+str(i+1)],orderB)

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



    @staticmethod
    def HTE(pathNum=12,outDegree=5):
        '''

        :param pathNum:
        :param outDegree:
        :return:
        '''
        data_dict = HTE.genData(pathNum=pathNum,outDegree=outDegree)
        VTree = data_dict['VTree']
        E = data_dict['E']
        S = data_dict['S']
        R = data_dict['R']


        # init
        V = R.copy()   ##新节点的集合
        V.append(0)
        I = []         ##每一轮 待处理的内节点集合
        Finish = False    ## 结束标志
        N = len(R) + 1
        V.append(N)
        newE = [(0, N)]        ##  new edge set
        for k in R:
            newE.append((N, k))
        I.append(N)
        N = N + 1

        while not Finish:
            dotI = []
            Finish = True
            for interNode in I:
                children = getChildren(newE, interNode)

                X = []   ##GMM 输入数据
                keyDict = []     ##  记录本次处理的目的节点
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


                ##  根据BIC准则得到component的个数
                BIC = np.infty
                for k in range(1, 4):
                    if len(X) < k:
                        continue
                    gmm = GMM(n_components=k, covariance_type='spherical')
                    gmm.fit(X)
                    bic = gmm.bic(X)
                    # print(bic)
                    if bic < BIC:
                        BIC = bic
                        best_gmm = gmm
                # print("components:", best_gmm.n_components)
                if best_gmm.n_components == 1:
                    continue
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
                Kp, Bp = HTE.progressive_search_Alg(means,dest_node,key_data)

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
                    weightedE= []
                    for node1 in VDict:
                        for node2 in VDict:
                            if node2 > node1:
                                w = HTE.calWOnSets(Ki[node1],Ki[node2],key_data,Bi)
                                weightedE.append((node1,node2,w))
                    ##HCS
                    G = nx.Graph()
                    G.add_weighted_edges_from(weightedE)
                    subgraph_list = []
                    div_factor = 2
                    HCS(G,subgraph_list,div_factor)
                    #得到HCS的结果 clusters
                    clusters = []
                    for subgraph in subgraph_list:
                        cluster = []
                        for item in subgraph:
                            cluster.extend(Ki[VDict[item]])
                        clusters.append(cluster)

                    #计算似然函数值
                    score3 = HTE.calL(clusters,S,gmm)
                    Lp.append((score3,clusters,Bi))
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
                    HTE.postmerge_Alg(hatK,hatL,hatB,key_data,S,gmm,betterResult)
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
                                    if (N,j) not in newE:
                                        newE.append((N, j))
                            if len(A) > 2:
                                dotI.append(N)
                                Finish = False
                            N = N + 1
            I = dotI
        inferredE = numberTopo(newE, R)
        data_dict['inferredE'] = inferredE
        # TreePlot(EtoVTree(inferredE))
        return data_dict


    @staticmethod
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
    @staticmethod
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
    @staticmethod
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
                    w = HTE.calWOnSets(cluster,other_cluster,key_data,hatB)
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
            score = HTE.calL(dotK,S,gmm)
            if score > hatL:
                betterResult.append([score,dotK])
                HTE.postmerge_Alg(dotK,score,hatB,key_data,S,gmm,betterResult)


    @staticmethod
    def doSim():
        PC = []     ## 精度
        ED = []     ## 编辑距离
        PN = [_ for _ in range(5, 15)]
        for pathNum in PN:
            cnt = 0
            edit_distance = []
            listObj = []
            for i in range(100):
                data = HTE.HTE(pathNum)
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
            with open('/home/zongwangz/文档/Projects/HTE/data/HTE_' + str(pathNum), 'w') as f:
                f.write(jsonObj)
                f.write('\n')
                f.write("mean edit distance:")
                f.write(str(np.mean(edit_distance)))
                f.write('\n')
                f.write("PC:")
                f.write(str(cnt/100))
                f.write('\n')
            ED.append(np.mean(edit_distance))
            PC.append(cnt / 100)

        fig1= plt.subplot()
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
    #test genData
    # HTE.genData()
    # test HTE
    # HTE.HTE()
    HTE.doSim()
    # HTE.test()
