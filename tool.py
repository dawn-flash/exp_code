import numpy as np
import random
import networkx as nx
import sys
from matplotlib import pyplot as plt
import scipy.stats as stats
import os
import ast  ## string list to list
def getChildren(E, parent):
    children = []
    for edge in E:
        if edge[0] == parent:
            children.append(edge[1])
    return children

def getParent(E, child):
    for edge in E:
        if edge[1] == child:
            return edge[0]
    return -1  # 没有父节点

def getDescendants(E, ancestor):
    descendants = []
    children = getChildren(E, ancestor)
    while children:
        parent = children[0]
        children1 = getChildren(E,parent)
        if children1:
            children.extend(children1)
        else:
            descendants.append(parent)
        del children[0]
    return descendants

def getAncestor(E, inode, dnode):
    child = inode
    parent = getParent(E, inode)
    while parent != -1:
        if isDescendant(E, dnode, parent):
            return parent
        else:
            parent = getParent(E, parent)
    # 无共同祖先节点
    return -1

def isDescendant(E, descendant, ancestor):
    child = descendant
    parent = getParent(E, child)
    while parent != -1:
        if ancestor == parent:
            return True
        else:
            parent = getParent(E, parent)
    return False

def Variance_way1(X):
    import numpy as np
    X = np.array(X)
    meanX = np.mean(X)
    n = np.shape(X)[0]
    # 按照协方差公式计算协方差，Note:分母一定是n-1
    variance = sum(np.multiply(X - meanX, X - meanX)) / (n-1)
    return variance

def GenTree(outDegree,pathNum):
    '''

    :param outDegree:
    :param pathNum:
    :return: VTree
    '''
    #linkSet=[0,1]
    linkSet = np.array([[0,1]])
    nodeNum=1
    #currentNodeSet=[1]
    currentNodeSet = np.array([[1]])
    flag=0
    pn=0;

    if outDegree > pathNum:
        print("出度要小于等于路径数\n")

    while pn < pathNum:
        #nextNodeSet = []
        nextNodeSet = np.array([[]])
        tempPN = pn
        addedNode = 0
        odegree = 0

        for i in range(len(currentNodeSet)):
            if currentNodeSet.size == 1:
                odegree = 2
            else:
                while True:
                    if pathNum - tempPN <3:
                        odegree = random.randint(1,3)
                    else:
                        odegree = random.randint(1,outDegree)
                    if tempPN+odegree+len(currentNodeSet)-i-1<=pathNum:
                        break;
            tempPN = tempPN+odegree

            if odegree > 1:
                flag = 1
                #childNodeSet =  range(nodeNum+1+addedNode,nodeNum+addedNode+odegree+1)
                childNodeSet = np.arange(nodeNum+1+addedNode,nodeNum+addedNode+odegree+1,1).reshape(1,odegree)
                if nextNodeSet.size == 0:
                    nextNodeSet = childNodeSet.T
                else:
                    nextNodeSet = np.r_[nextNodeSet,childNodeSet.T]
                tempNodeSet1 = np.tile(currentNodeSet[i,:],(odegree,1))
                tempNodeSet2=np.c_[tempNodeSet1,childNodeSet.T]
                linkSet = np.r_[linkSet, tempNodeSet2]
                #linkSet = np.r_[linkSet,np.c_[np.tile(currentNodeSet[i,:],(odegree,1)),childNodeSet.T]]
                addedNode = addedNode + odegree
            else:
                pn=pn+1
        if not flag:
            flag = 0;
            pn = pn - currentNodeSet.size;
            continue
        else:
            flag=0
            nodeNum = nodeNum +addedNode
            currentNodeSet=nextNodeSet;
        if pathNum-currentNodeSet.size == pn:
            pn=pathNum
    VTree=linkSet[:,0]
    return VTree



def getInternalNodes(VTree):
    internalNodes = []
    for i in VTree:
        internalNodes.append(i)
    return internalNodes

def getLeafNodes(VTree):
    '''

    :param VTree:
    :return: leafNodes
    '''
    leafNodes = []
    for i in range(len(VTree)):
        leafNode = i+1
        if leafNode not in VTree:
            leafNodes.append(leafNode)
    return leafNodes


def gen_linkDelay(VTree,scale = 1.0,probesNum = 200):
    '''
    产生链路时延
    :return:
    '''

    linkDelay = []
    for i in range(len(VTree)):
        linkDelay.append(np.random.exponential(scale, probesNum))
    return linkDelay

def calPathDelay(VTree, linkDelay):
    '''
    计算路径时延
    :return:
    '''
    PathDelay = []
    for i in range(len(VTree)):
        if i+1 not in VTree:
            tempsum = linkDelay[i]
            j = i+1
            while VTree[j-1] != 0:
                j = VTree[j-1]
                tempsum =tempsum+linkDelay[j-1]
            PathDelay.append(tempsum)
    return PathDelay

def Covariance_way2(X,Y):
    '''
    向量中心化方法计算两个等长向量的协方差convariance
    '''
    X, Y = np.array(X), np.array(Y)
    n = np.shape(X)[0]
    centrX = X - np.mean(X)
    centrY = Y - np.mean(Y)
    convariance = sum(np.multiply(centrX, centrY)) / (n - 1)
    # print('向量中心化方法求得协方差:', convariance)
    return convariance

def VTreetoE(VTree):
    E = []
    child = 1
    for parent in VTree:
        E.append((parent, child))
        child = child + 1
    return E

def numberTreeOld(E, root='s'):
    assert isinstance(E, list)
    newE = []  # 新的边list
    number = 0  # 正在处理的节点
    count = 1  # 已经出现的节点个数
    parents = [root]  # 待处理的节点
    while parents:
        parent = parents.pop()  # 将要处理的节点
        newParent = number  # 该节点的新的编号
        children = getChildren(E, parent)
        parent = number
        for child in children:
            parents.insert(0, child)
            newE.append((newParent, count))
            count = count+1
        number = number+1
        if count == len(E)+1:  # 当出现的节点个数符合要求就可以终止处理了
            break
    return newE



def toBracketString(sourceE, E):
    '''
    将VTree和NJ算法输出的包含边的list转换为apted算法所需要的BracketString模式
     :param VTree: [0 1 1 2 2 4 4 4]  [(0,1], (1,2), (1,3), (2,4), (2,5), (4,6), (4,7), (4,8)]
     :param E:[(5, 2), (5, 4), (5, 3), (6, 1), (6, 5), (7, 0), (7, 6), ('s', 7)]
     :return: t1:{0{1{2{4{6}{7}{8}}{5}}{3}}} t2:{s{7{6{5{3}{4}{2}}{1}}{0}}}
               t2:
    '''
    t1 = '{0}'
    indexes = [[0, 1]]
    while indexes:
        index = indexes.pop() # 顺序存储，逆序取出处理，这样字符串就不会乱
        children = getChildren(sourceE, index[0])
        if len(children) == 0:
            continue
        offset = 1+len(str(children[0]))
        circle = 0
        for i in range(len(children)):
            t1 = t1[:index[1]+circle+1]+'{'+'{0}'.format(children[i])+'}'+t1[index[1]+circle+1:]
            indexes.append([children[i], index[1]+offset])
            circle = circle+2+len(str(children[i]))
            if i+1 == len(children):
                continue
            else:
                offset = offset+2+len(str(children[i+1]))

    t2 = '{0}'
    indexes2 = [[0, 1]]
    while indexes2:
        index2 = indexes2.pop() # 顺序存储，逆序取出处理，这样字符串就不会乱
        children2 = getChildren(E, index2[0])
        if len(children2) == 0:
            continue
        offset2 = 1+len(str(children2[0]))
        circle2 = 0
        for i in range(len(children2)):
            t2 = t2[:index2[1]+circle2+1]+'{'+'{0}'.format(children2[i])+'}'+t2[index2[1]+circle2+1:]
            indexes2.append([children2[i], index2[1]+offset2])
            circle2 = circle2+2+len(str(children2[i]))
            if i+1 == len(children2):
                continue
            else:
                offset2 = offset2+2+len(str(children2[i+1]))

    # print(t1)
    # print(t2)
    return t1,t2


def getRM(D, VTree):
    '''
    产生路由矩阵
    :param D:[4,5,6,7,8]
    :param VTree:[0,1,1,2,2,2,3,3]
    :return:
    '''
    pathNum = len(D)
    linkNum = len(VTree)
    A = np.zeros((pathNum,linkNum))
    for i in range(pathNum):
        leafNode = D[i]
        A[i][leafNode-1] = 1
        parentNode = VTree[leafNode-1]
        while parentNode != 0:
            A[i][parentNode-1] = 1
            parentNode = VTree[parentNode-1]
    return A

def numberTopo(E,R):
    '''
    根据路径编号
    :param E:
    :param R:
    :return:newE
    '''
    transfer = {} ##内部节点转换表
    newE = []  ##具有一定顺序的边集合
    number = len(R)+1  ##编号
    for leafnode in R:
        trace = []  #一条路径的轨迹
        parent = getParent(E,leafnode)
        trace.append(parent)
        cnt = 0
        while getParent(E,parent) != 0:
            cnt += 1
            if cnt > 100:
                print("死循环",E)
                sys.exit(0)
            parent = getParent(E,parent)
            trace.append(parent)
        trace.reverse()
        for i in range(len(trace)):
            if trace[i] not in transfer:
                transfer[trace[i]] = number
                number = number+1
                trace[i] = transfer[trace[i]]
            else:
                trace[i] = transfer[trace[i]]
        tempE = []
        if (0,trace[0]) not in newE:
            tempE.append((0,trace[0]))
        if len(trace ) == 1:
            if (trace[0],leafnode) not in newE:
                tempE.append((trace[0],leafnode))

        else:
            index = 0
            while index < len(trace)-1:
                if (trace[index], trace[index+1]) not in newE:
                    tempE.append((trace[index], trace[index+1]))
                index = index+1
            if (trace[len(trace)-1],leafnode) not in newE:
                tempE.append((trace[len(trace)-1],leafnode))
        tempE.reverse()
        newE.extend(tempE)
    return newE

def numberTopoByVTree(VTree,root = 0):
    '''

    :param VTree:[0,1,1,1,2,2]
    :param root:
    :return: edge set
    '''
    # leafNodes = getLeafNodes(VTree) ##需要调整顺序
    leafNodes = []
    E = VTreetoE(VTree)
    visit = []
    visit.append(root)
    while visit:
        node = visit[0]
        del visit[0]
        children = getChildren(E,node)
        if len(children) == 0:
            leafNodes.append(node)
        for i in range(len(children)):
            visit.insert(i,children[i])
    n = len(leafNodes)
    R = np.arange(1,n+1)
    transfer = {}  ##内部节点转换表
    newE = []  ##具有一定顺序的边集合
    number = n + 1  ##编号
    for i in range(n):
        trace = []  # 一条路径的轨迹
        parent = getParent(E, leafNodes[i])
        trace.append(parent)
        while getParent(E, parent) != root:
            parent = getParent(E, parent)
            trace.append(parent)
        trace.reverse()
        for j in range(len(trace)):
            if trace[j] not in transfer:
                transfer[trace[j]] = number
                number = number + 1
                trace[j] = transfer[trace[j]]
            else:
                trace[j] = transfer[trace[j]]
        tempE = []
        if (0, trace[0]) not in newE:
            tempE.append((0, trace[0]))
        if len(trace) == 1:
            if (trace[0], R[i]) not in newE:
                tempE.append((trace[0], R[i]))
        else:
            index = 0
            while index < len(trace) - 1:
                if (trace[index], trace[index + 1]) not in newE:
                    tempE.append((trace[index], trace[index + 1]))
                index = index + 1
            if (trace[len(trace) - 1], R[i]) not in newE:
                tempE.append((trace[len(trace) - 1], R[i]))
        tempE.reverse()
        newE.extend(tempE)
    return newE


def EtoVTree(E):
    i = 1
    VTree = []
    while getParent(E,i) != -1:
        parent = getParent(E,i)
        VTree.append(parent)
        i = i+1
    return VTree


def getSharedPathLenbyNodes(E, iNode, jNode):
    ancestor = getAncestor(E, iNode, jNode)
    parent = getParent(E, ancestor)
    len = 1
    while parent != 0:
        parent = getParent(E, parent)
        len = len+1
    return len

def getSharePathDelaybyNodes(E, linkDelay,iNode,jNode):
    '''
    专用
    :return:
    '''
    ancestor = getAncestor(E, iNode, jNode)
    delay = linkDelay[ancestor-1]
    parent = getParent(E, ancestor)
    while parent != 0:
        delay = delay+linkDelay[parent-1]
        parent = getParent(E, parent)
    return delay

def getLinkNumBetweenNodes(E, iNode, jNode):
    len = -1
    if isDescendant(E, iNode, jNode):
        len = 1
        parent = getParent(E, iNode)
        while parent != jNode:
            len = len+1
            parent = getParent(E, parent)
        return len
    elif isDescendant(E, jNode, iNode):
        len = 1
        parent = getParent(E, jNode)
        while parent != iNode:
            len = len+1
            parent = getParent(E, parent)
        return len
    else:
        ancestor = getAncestor(E, iNode, jNode)
        len = getLinkNumBetweenNodes(E, iNode, ancestor)
        len = len+getLinkNumBetweenNodes(E, jNode, ancestor)
    return len


def plot_Tree(VTree,link_state): #画出树型拓扑
    import matplotlib.pyplot as plt
    plt.figure() # 新开一个作图窗口

    # link = linkSetVec[:, 0]
    link = VTree
    congestedLink = link_state

    x,y=treeLayout(link) #调用树布局函数
    n=len(link)
    i=0
    for f in link:
        if f==0:
            i+=1
    leaves=[]
    while i<n+1:
        b=0
        j=0
        while j<n:
            if i==link[j]: #如果有了第一个父节点为i的点就跳出循环
                b=j
                break
            j+=1
        if b==0: #如果没有找到父节点为i的节点，就为叶子节点
            leaves.append(i)
        i+=1

    num_layers=1/min(y)-1
    num_layers=int(num_layers+0.5)

    i=0
    chains=[]
    while i<len(leaves): #判断链路并加入到chains队列中
        index=leaves[i]-1
        chain=[]
        chain.append(index + 1)
        parent_index=link[index]-1
        j=1
        while parent_index != 0:
            chain.append(parent_index+1)
            parent_index=link[parent_index]-1
            j += 1
        chain.append(1)
        chain.reverse()
        chains.append(chain)
        i += 1

    y_new=y
    y_new=np.zeros(len(y))
    i = 0
    while i<len(y): #调整y的值
        r=0
        j=0
        b=0
        while j<len(leaves):
            r=0
            for c in chains[j]:
                if c==i+1:
                    b=1
                    break
                elif c!=i+1:
                    r += 1

            if b==1:
                break
            elif b==0:
                j += 1
        y_new[i]=0.9-(r-1)/(num_layers+1)
        i += 1
    plt.figure()
    #画线
    i=0
    while i<len(leaves):
        j=0
        while j+1<len(chains[i]):
            line_x=[]
            line_y=[]
            line_x.append(x[chains[i][j]-1])
            line_x.append(x[chains[i][j+1]-1])
            line_y.append(y_new[chains[i][j]-1])
            line_y.append(y_new[chains[i][j+1]-1])
            if j+2==len(chains[i]) and congestedLink[chains[i][j+1]-1]==0:
                plt.plot(line_x,line_y,'g-',linewidth=0.5)
            elif j+2==len(chains[i]) and congestedLink[chains[i][j+1]-1]>0:
                plt.plot(line_x,line_y,'r-',linewidth=0.5)
            elif congestedLink[chains[i][j+1]-1]==0:
                plt.plot(line_x,line_y,'g-')
            elif congestedLink[chains[i][j+1]-1]>0:
                plt.plot(line_x,line_y,'r-')
            j += 1
        i += 1
    #画点
    i=0
    while i<len(leaves):
        j=0
        while j<len(chains[i]):
            point_x=[]
            point_y=[]
            point_x.append(x[chains[i][j]-1])
            point_y.append(y_new[chains[i][j]-1])
            if j+1==len(chains[i]):#画出叶子节点
                plt.plot(point_x,point_y,'bo',linewidth=0.5)
            else:#画出非叶子节点
                plt.plot(point_x,point_y,'ko')
            j += 1
        i += 1

    for i in range(len(x)): #加上标号
        plt.text(x[i]*1.02, y_new[i]*1.02, str(i+1))

    plt.plot(x[-1],y_new[-1],'bo') # 'Destination Node'
    plt.plot(x[0],y_new[0],'ko') # 'Internal Node'
    plt.plot(x[0:1],y_new[0:1],'g-')
    plt.plot(label='line')
    xx=[x[0],x[0]]
    yy=[0.9-(-2)/(num_layers+1),y_new[0]]
    if congestedLink[0]==0:
        plt.plot(xx,yy,'g-')
    else:
        plt.plot(xx,yy,'r-')
    plt.plot(xx[0],yy[0],'k*') #'Root Node'
    plt.text(xx[0]*1.05,yy[0],str(0),family='serif',style='italic',ha='right',wrap=True)

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

def treeLayout(parent): #生成树型拓扑中节点的坐标(x,y)
    #Lay out tree or forest
    #parent is the vector of parent pointers,with 0 for a root
    #post is a postorder permutation on the tree nodes
    #xx and yy are the vector of coordinates in the unit square at which
    #to lay out the nodes of the tree to make a nice picture
    #Optionally h is the height of the tree and s is the number of vertices
    #in the top-level separator
    pv=[]
    n=len(parent)
    parent,pv=fixparent(parent)

    #j=find(parent) in matlab
    j=np.nonzero(parent)
    jj=[x+1 for x in j[0]]

    #A=sparse(parent(j),j,1 n,n); in matlab
    A=np.zeros((n,n))
    for i in range(len(jj)):
        A[parent[i]-1,jj[i]-1]=1
    A=A+A.T+np.eye(n)
    post=etree(A)

    #Add a dummy root node and identify the leaves
    for _ in range(len(parent)):
        if parent[_]==0:
            parent[_]=n+1       #change all 0s to n+1s

    #in postorder computer height and descendant leaf intervals
    #space leaves evenly in x
    isaleaf = [1 for _ in range(n+1)]
    for i in range(len(parent)):
         isaleaf[parent[i]-1]=0

    xmin = [n for _ in range(len(parent)+1)]
    xmax = [0 for _ in range(len(parent)+1)]
    height=[0 for _ in range(len(parent)+1)]
    nkids =[0 for _ in range(len(parent)+1)]
    nleaves = 0

    for i in range(1,n+1):
        node = post[i-1]
        if isaleaf[node-1]:
            nleaves = nleaves + 1
            xmin[node-1] = nleaves
            xmax[node-1] = nleaves

        dad = parent[node-1]
        height[dad-1] = max(height[dad-1],height[node-1]+1)
        xmin[dad-1] = min(xmin[dad-1],xmin[node-1])
        xmax[dad-1] = max(xmax[dad-1],xmax[node-1])
        nkids[dad-1] = nkids[dad-1] + 1

    #compute coordinates leaving a little space on all sides
    treeht = height[n] - 1
    deltax = 1/(nleaves+1)
    deltay = 1/(treeht+2)
    x=[]
    y=[]

    #Omit the dummy node
    for _ in range(len(xmin)):
        x.append(deltax*(xmin[_]+xmax[_])/2)
    for _ in range(len(height)):
        y.append(deltay*(height[_]+1))

    for i in range(-1,-1*len(nkids)):
        if nkids[i]!=1:
            break
    xx=[]
    yy=[]
    flagx=1
    flagy=1

    for _ in  pv:
       for i in range(len(pv)):
           if pv[i]==flagx:
               xx.append(x[i])
               flagx=flagx+1
       for i in range(len(pv)):
           if pv[i]==flagy:
               yy.append(y[i])
               flagy=flagy+1


    return xx,yy

def fixparent(parent):
    # Fix order of parent vector
    # [a,pv]= fixparent(B) takes a vector of parent nodes for an elimination
    # tree, and re-orders it to produce an equivalent vector
    # a in which parent nodes are always higher-number than child nodes
    # if B is an elimination tree produced by the tree funtion, this step will not
    # be necessary. PV is a permutation vector,so that A=B(pv)
    n=len(parent)
    a=parent
    a[a==0] = n+1
    pv = [x for x in range(1,n+1) ]
    niter = 0
    while(True):
        temp=[_ for _ in range(1,n+1)]
        x=[]
        for i in range(len(a)):
            if (a[i]<temp[i]):
                x.append(i+1)
            else:
                x.append(0)

        k=np.nonzero(x)
        kk=[xx+1 for xx in k]
        if len(kk[0])==0:
            break
        kk=kk[0][0]
        j=a[kk-1]

        a=a.tolist()
        tem=a[kk-1]
        del(a[kk-1])
        a.insert(j-1,tem)
        a=np.array(a)

        tem=pv[kk-1]
        del(pv[kk-1])
        pv.insert(j-1,tem)

        te = [0 for _ in range(len(a))]
        for _ in range(len(a)):
            if j<=a[_]<kk:
                te[_]=1
        for _ in range(len(a)):
            if a[_]==kk:
                a[_]=j
        for _ in range(len(te)):
            if te[_]==1:
                a[_]=a[_]+1
        niter = niter + 1


    a[a>n] = 0
    return a,pv


def etree(mat):
    ##为其上三角形是 A 的上三角形的对称方阵返回消去树？？？
    if mat.shape[0]==mat.shape[1]:
        return [x for x in range(1,mat.shape[0]+1)]
    else:
        pass





def TreePlot(VTree):
    '''
    nodes = np.array(VTree).reshape((1, len(VTree)))
    :param args:
    :return:
    '''
    E = VTreetoE(VTree)
    N = len(VTree)+1  ## 节点数目

    ## x的坐标 初始值为-1
    x = []
    for i in range(N):
        x.append(-1)
    x = np.array(x, dtype=float)
    ## 先设置x的坐标
    leafNodes = getLeafNodes(VTree)
    n = len(leafNodes)
    for i in range(n):
        x[i+1] = i
    ## 接着设置内节点
    for i in range(n+1, N):  # 等于孩子节点的坐标均值
        if x[i] == -1:
            parent = i
            children = getChildren(E, parent)
            sum = []
            for child in children:
                if x[child] == -1:
                    to_setXi(child,x,E)
                    sum.append(x[child])
                else:
                    sum.append(x[child])
            x[i] = np.mean(sum)
    ##设置根结点
    to_setXi(0,x,E)
    ##设置x的绝对位置：
    nn = n+2
    for i in range(N):
        x[i] = x[i]+1
    x_x = 1/nn
    for i in range(N):
        x[i] = x[i]*x_x

    ##设置y的坐标
    y = []
    for i in range(N):
        y.append(-1)
    y = np.array(y,dtype=float)
    layer = 0  ## 层数
    flag = True
    layer_visit = [0]
    while flag:
        next_layer = []
        while len(layer_visit) != 0:
            parent = layer_visit[0]
            y[parent] = layer
            del layer_visit[0]
            children = getChildren(E, parent)
            if len(children) != 0:
                next_layer.extend(children)
        layer = layer+1
        if len(next_layer) == 0:
            flag = False
        else:
            layer_visit = next_layer
    max_layer = np.max(y)
    ## 新把层数颠倒过来
    for i in range(N):
        y[i] = max_layer-y[i]
    # 预留空间 所以上下留一层
    max_layer = max_layer+2
    for i in range(N):
        y[i] = y[i]+1
    # 设置绝对位置
    y_y = 1/max_layer
    for i in range(N):
        y[i] = y[i]*y_y
    #画图
    plt.scatter(x,y,c='r',marker = 'o')
    for edge in E:
        parent = edge[0]
        child = edge[1]
        plt.plot([x[parent],x[child]],[y[parent],y[child]],color='b')
    for node in range(N):
        plt.annotate(s=str(node),xy=(x[node],y[node]),xytext=(x[node]+0.010,y[node]))
    plt.show()

def to_setXi(node,x,E):
    parent = node
    children = getChildren(E, parent)
    sum = []
    for child in children:
        if x[child] == -1:
            to_setXi(child, x, E)
            sum.append(x[child])
        else:
            sum.append(x[child])
    temp = np.mean(sum)
    x[node] = np.mean(sum)

def HCS(G,subgraph_list,div_factor):
    if len(G.nodes()) <= 2:
        nodes = []
        for node in G.node:
            nodes.append(node)
        if len(nodes) == 1:
            subgraph_list.append([nodes[0]])
            return
        elif len(nodes) == 2:
            node1 = nodes[0]
            node2 = nodes[1]
            if G.edge[node1][node2]['weight'] < 1/2:
                subgraph_list.append([node1])
                subgraph_list.append([node2])
                return
            else:
                subgraph_list.append([node1,node2])
                return
        else:
            return
    random_node_1 = random.randrange(0, len(G.nodes()), 1)
    random_node_2 = random.randrange(0, len(G.nodes()), 1)
    while random_node_2 == random_node_1:
        random_node_2 = random.randrange(0, len(G.nodes()), 1)

    cut_val, partition = nx.minimum_cut(G, G.nodes()[random_node_1], G.nodes()[random_node_2], capacity='weight')

    G1 = nx.subgraph(G, partition[0])  # subgraph 1
    G2 = nx.subgraph(G, partition[1])  # subgraph 2

    if cut_val >= len(G.nodes()) / div_factor:
        nodes = []
        for node in G.node:
            nodes.append(node)
        subgraph_list.append(nodes)
        return
    else:
        HCS(G1, subgraph_list, div_factor)
        HCS(G2, subgraph_list, div_factor)
        return

def to_zzzNode(E,root=0):
    from zss import Node
    A = Node(str(root))
    U = [(0,A)]
    while len(U) != 0:
        parent = U[0][0]
        node = U[0][1]
        del U[0]
        children = getChildren(E, parent)
        if len(children) == 0:
            continue
        for i in range(len(children)):
            node.addkid(Node(str(children[i])))
            U.append((children[i], node.children[i]))
    return A

def calEDbyzss(E1,E2,root=0):
    '''
    计算编辑距离
    :param E1:
    :param E2:
    :param root:
    :return:
    '''
    from zss import simple_distance
    A1 = to_zzzNode(E1)
    A2 = to_zzzNode(E2)
    return simple_distance(A1,A2)



'''
tool for ns3
author: zzw
time: 2018.10.30
'''
def GenVTrees(pathNumList=[i+5 for i in range(9)],num_VTree=100,outDegree=5):
    '''
     生成指定数目和参数的VTree，存入文件中
    :param pathNumList:
    :param num_VTree:
    :param outDegree:
    :return:
    '''
    filename = "VTrees" + "_" + str(outDegree) + "_" + str(pathNumList[0]) + "_" + str(pathNumList[-1])
    VTrees = []
    for pathNum in pathNumList:
        for i in range(num_VTree):
            VTree0 = GenTree(outDegree,pathNum)
            E = numberTopoByVTree(VTree0)
            VTree = EtoVTree(E)
            # TreePlot(VTree)
            VTrees.append(VTree)
            with open(filename,'a+') as f:
                f.write(str(VTree))
                f.write('\n')

def getVTrees(filename="/home/zongwangz/PycharmProjects/Topo_4_3_10"):
    '''
    从VTree_5_5_13中获取所有的VTree
    :return:
    '''
    VTrees = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            VTree = line[1:-2].split(',')
            for item in VTree:
                VTree[VTree.index(item)] = int(item)
            VTrees.append(VTree)
    return VTrees

if __name__ == "__main__":
    pass
