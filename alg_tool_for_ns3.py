'''
this is the tool for ns3 to test alg
author: zongwang.zhang
mail:zongwang.zhang@outlook.com
'''
from tool import *  #tool for algorthms
import time
import re
from sim_tool import *
def genTopo(PathNumList=[3,4,5,6,7,8,9,10],MaxOutDegree=4,num_VTree=100):
    '''
    to generate total 800 topologies for testing the algorthms.
    :param PathNumList:
    :param MaxOutDegree:
    :param num_VTree:
    :return:
    '''
    filename = "Topo" + "_" + str(MaxOutDegree) + "_" + str(PathNumList[0]) + "_" + str(PathNumList[-1])
    VTrees = []
    for pathNum in PathNumList:
        if pathNum < MaxOutDegree:
            outDegree = pathNum
        else:
            outDegree = MaxOutDegree
        for i in range(num_VTree):
            VTree0 = GenTree(outDegree,pathNum) # like [0,1,1,1]
            E = numberTopoByVTree(VTree0)   #renumber the VTrees E=[(4,1),(4,2),(4,3),0,4)]
            VTree = EtoVTree(E)   # the new VTree is [4,4,4,0]
            # TreePlot(VTree)   #plot the Tree
            VTrees.append(VTree)
            with open(filename,'a+') as f:
                f.write(str(VTree))
                f.write('\n')

def calMetrics(calmetric,saveS,pktnum=200):
    '''
    to calculate the average metric
    :param: calmetric   the function to calculate metrics
    :param pktnum:
    :return:
    '''
    VTrees = getVTrees("/home/zongwangz/PycharmProjects/Topo_4_3_10")
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
            f3 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/order"+str(i)
            S = calmetric(f1,f3,VTrees[i])
            saveS(S,f2)
            os.remove(f1)
            os.remove(f3)
def saveS1(S,filename):
    '''
    just to save the file for sandwich metric
    :param S:
    :param filename:
    :return:
    '''
    if os.path.exists(filename):
        os.remove(filename)
    for i in range(len(S[0])):
        for j in range(len(S[0])):
            if len(S[i][j]) != 0:
                open(filename,"a+").write(str(S[i][j]))
                open(filename, "a+").write("\n")

def saveS2(S,filename):
    '''
    just to save the file for back to back path delay or loss
    :param S:
    :param filename:
    :return:
    '''
    for key in S:
        open(filename,"a+").write(str(S[key]))
        print(str(S[key]))
        open(filename, "a+").write("\n")



def calmetric1(filename,VTree,pktnum=200):
    '''
    to calculate interarrival time difference of sandwich probing from single trace file
    the data structure：
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
    S = [[ [] for j in range(n)] for i in range(n)]
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
                        sys.exit(0)
                    if sandwich_len == 0:
                        #正在记录三明治包的第一个报文，验证
                        if size == 50:
                            sandwich.append([packet_id,dest_node])
                        else:
                            print("first packet in sandwich packet lost in 0 node!!")
                            sys.exit(0)
                    if sandwich_len == 1:
                        ##正在记录三明治包的第二报文
                        if size == 1400:
                            sandwich.append([packet_id,dest_node])
                        else:
                            print("second packet in sandwich packet lost in 0 node!!")
                            sys.exit(0)
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
                            sys.exit(0)
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
                                        sys.exit(0)
                                    if index == 1 and item[2] == packet_id:
                                        index = 2
                                        item[index] = time
                                        if isinstance(item[0], float) and isinstance(item[1], float) and isinstance(
                                                item[2], float):
                                            arrivalInterval = item[2] - item[0]
                                            S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
                                            toRemove.append(key)
                                            toRemove.append(item)
                                        flag1 = False
                                        break
                                    if index == 0 or index == 2:
                                        item[index] = time
                                        if isinstance(item[0],float) and isinstance(item[1],float) and isinstance(item[2],float):
                                            arrivalInterval = item[2]-item[0]
                                            S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
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
                                            S[int(key.split(",")[0])-1][int(key.split(",")[1])-1].append(arrivalInterval)
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
                        print(toRemove[1],cnt)
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

def calmetric2(filename,filename2,VTree,pktnum=2000):
    '''
    to caculate link delay of back to back probing from single trace file
    the data structure：
    {
        "1,2":{
            "packet_order":[[packet_id,  ]
            ]
        }
    }
    :param filename:
    :param filename2:
    :param VTree:
    :param oktnum:
    :return:
    '''
    ## 先处理一下 记录发包顺序的文件
    f = open(filename2,"r")
    lines = f.readlines()
    R = getLeafNodes(VTree)  ##保存叶子节点
    n = len(R)
    streamID = [0 for _ in range(n)]## 记录每一个目的节点的包id，相当于一条流上面的id
    Orders = []  ## 根据发包顺序记录包id
    Time = []  ## 与order对应的发包时间
    for line in lines: ##处理记录发包顺序的文件，将节点对编号转换为，包id的编号
        str1,str2,str3,str4 = line.split(",")
        id1 = int(str1)
        id2 = int(str2)
        time1 = float(str3)
        time2 = float(str4)
        Orders.append([streamID[id1-1],streamID[id2 - 1]])
        Time.append([time1,time2])
        streamID[id1-1] = streamID[id1-1]+1
        streamID[id2 - 1] = streamID[id2 - 1] + 1

    ##根据仿真代码的设置规则，设定源ip和目的ip
    srcIP = "10.1." + str(n + 1) + ".1"
    destIPList = []
    for item in R:
        destIPList.append("10.1." + str(item) + ".2")

    ##保存计算出来数据
    S = {}
    for i in R:
        for j in R:
            if j > i:
                key = str(i) + "," + str(j)
                S[key] = []


    ##用于记录所有包
    record = {}
    sum = (n - 1 + 1) / 2 * (n - 1) * pktnum
    cnt = 0
    for i in R:
        for j in R:
            if j > i:
                key = str(i) + "," + str(j)
                record[key] = {}
                record[key]["packet_order"] = []
                record[key]["num"] = 0

    b2b = []    ## 记录一个背靠背包

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                oc = re.split(r"\)|\(| ", line)
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
                if currentNode == 0 and src_ip == srcIP and dest_ip in destIPList and action == "+":
                    dest_node = int(dest_ip.split('.')[2])
                    ##记录根节点流出的背靠背包流
                    b2b_len = len(b2b)
                    if b2b_len == 0:
                        # 正在记录三明治包的第一个报文，验证
                        if size == 50:
                            b2b.append([packet_id, dest_node, time]) ## 虚假的time
                        else:
                            print("first packet in b2b packet lost in 0 node!!")
                            sys.exit(0)
                    if b2b_len == 1:
                        ##正在记录三明治包中的第三个报文
                        if size == 50 and [b2b[0][0],packet_id] in Orders:
                            ## 每一对包id都是唯一的，所以可以根据这个特性来判断是否从属与同一个背靠背包
                            ##是否有丢包
                            index = Orders.index([b2b[0][0],packet_id])
                            b2b.append([packet_id, dest_node,time]) ##虚假的time
                            key = str(b2b[0][1]) + "," + str(dest_node)
                            record[key]["packet_order"].append([ [b2b[0][0],Time[index][0]], [packet_id,Time[index][1]],[0,0]]) ## 真正的time
                            record[key]["num"] = record[key]["num"] + 1
                            b2b = []
                            del Orders[index]
                            del Time[index]
                        elif size == 50 and [b2b[0][0],packet_id] not in Orders:
                            ## 说明这个背靠背包已经丢失,有两种情况，一种是第一个报文丢失或者是 第二个报文丢失
                            ## 如果此时b2b中已经记录的是第二个包，目前正在检查的是其他背靠背包中的第一个(或者第二个)报文
                            ## 此时需要 舍去b2b, 并且把正在检查的装入b2b
                            ## 如果此时b2b记录的是第一个包，那个正在检查的是其他背靠背包中的第一个(或者第二个)报文，执行
                            ##的操作和上面一致
                            print("packet in b2b packet lost in 0 node!!")
                            b2b = []
                            b2b.append([packet_id, dest_node,time])
                if currentNode in R and src_ip == srcIP and dest_ip in destIPList and action == "r":
                    ##接受背靠背包中的报文，有可能是第一个报文，也可能是第二个报文。由它的id和目的节点可以唯一找到一个位置，
                    ## 存入该报文的time
                    toRemove = []
                    dest_node = int(dest_ip.split('.')[2])
                    ##当前节点为接受节点
                    if size == 50:
                        ##此时接受报文,找到这个报文可能的位置
                        flag1 = False
                        for key in record:
                            ##可能的位置有两个，比如“1,2”或者“2,3”，最后得到的结果只有一个位置
                            if int(key.split(",")[0]) == dest_node:
                                ##为“2,3”这种情况，
                                for item in record[key]["packet_order"]:
                                    if packet_id == item[0][0]:
                                        item[0][1] = time-item[0][1]
                                        item[2][0] = 1
                                        if item[2][0] == 1 and item[2][1] == 1:
                                            ##这个背靠背包组装完毕，装入S
                                            S[key].append([item[0][1],item[1][1]])
                                            toRemove.append(key)
                                            toRemove.append(item)
                                        flag1 = True
                                        break
                                    else:
                                        continue
                            elif int(key.split(",")[1]) == dest_node:
                                ##为“1,2”这种情况，
                                for item in record[key]["packet_order"]:
                                    if packet_id == item[1][0]:
                                        item[1][1] = time - item[1][1]
                                        item[2][1] = 1
                                        if item[2][0] == 1 and item[2][1] == 1:
                                            ##这个背靠背包组装完毕，装入S
                                            S[key].append([item[0][1], item[1][1]])
                                            toRemove.append(key)
                                            toRemove.append(item)
                                        flag1 = True
                                        break
                                    else:
                                        continue
                            if flag1:
                                break
                    if len(toRemove) == 2:
                        record[toRemove[0]]["packet_order"].remove(toRemove[1])
                        cnt = cnt + 1
                        print(toRemove[0],toRemove[1], cnt)
                        if cnt == sum:
                            print("all receive")
                            break
                        if cnt == sum - 1:
                            pass
            else:
                break
    '''
    除输出外剩余的丢失情况在record中，num参数也可以进行操作来得知丢失情况 这里时间有限，没有写这部分，但是结果中考虑了丢失情况
    '''
    return S

def calmetric3(filename, filename2, VTree, pktnum=2000,):
    '''
    to caculate success transmit rate of back to back probing from single trace file
    the data structure：
    {
        "1,2":{
            "packet_order":[[packet_id,  ]
            ]
        }
    }
    :param filename:
    :param filename2:
    :param VTree:
    :param pktnum:
    :return:
    '''
    R = getLeafNodes(VTree)  ##保存叶子节点
    n = len(R)
    Orders = {}
    for i in R:
        for j in R:
            if j > i:
                key = str(i) + "," + str(j)
                Orders[key] = []

    ## 先处理一下 记录发包顺序的文件
    f = open(filename2, "r")
    lines = f.readlines()
    streamID = [0 for _ in range(n)]  ## 记录每一个目的节点的包id，相当于一条流上面的id
    for line in lines:  ##处理记录发包顺序的文件，将节点对编号转换为，包id的编号
        str1, str2 = line.split(",")
        id1 = int(str1)
        id2 = int(str2)
        key = str(id1)+","+str(id2)
        Orders[key].append([streamID[id1 - 1], streamID[id2 - 1],0,0])## 前面两个位置是包id，后面两个位置是是否接受
        streamID[id1 - 1] = streamID[id1 - 1] + 1
        streamID[id2 - 1] = streamID[id2 - 1] + 1
    '''
    接收包的顺序也必然如此,否则丢包
    Orders = {
    "1,2":[[0,0,0,0],[1,2,0,0]],
    "1,3":[],
    "2,3":[[1,0,0,0]]
    }
    '''

    ##根据仿真代码的设置规则，设定源ip和目的ip
    srcIP = "10.1." + str(n + 1) + ".1"
    destIPList = []
    for item in R:
        destIPList.append("10.1." + str(item) + ".2")

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                oc = re.split(r"\)|\(| ", line)
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
                if currentNode in R and src_ip == srcIP and dest_ip in destIPList and action == "r":
                    ##接受背靠背包中的报文，有可能是第一个报文，也可能是第二个报文。由它的id和目的节点可以唯一找到一个位置，
                    dest_node = int(dest_ip.split('.')[2])
                    ##当前节点为接受节点
                    if size == 50:
                        ##此时接受报文,找到这个报文可能的位置
                        flag1 = False
                        for key in Orders:
                            ##可能的位置有两个，比如“1,2”或者“2,3”，最后得到的结果只有一个位置
                            if int(key.split(",")[0]) == dest_node:
                                ##为“2,3”这种情况，
                                for item in Orders[key]:
                                    if packet_id == item[0]:
                                        item[2] = 1
                                        flag1 = True
                                        break

                            elif int(key.split(",")[1]) == dest_node:
                                ##为“1,2”这种情况，
                                for item in Orders[key]:
                                    if packet_id == item[1]:
                                        item[3] = 1
                                        flag1 = True
                                        break
                                    else:
                                        continue
                            if flag1:
                                break
            else:
                break

    return Orders
def genSourceEs(data_dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/light_load"):
    '''
    生成sourceE文件
    :param filename:
    :return:
    '''
    c = 1
    FILENAME = data_dir+"/SourceE"
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
    f = open(FILENAME,"a+")
    filename = data_dir+"/Topo_4_3_10"
    VTrees = getVTrees(filename)
    for VTree in VTrees:
        sourceE = numberTopo(VTreetoE(VTree), getLeafNodes(VTree))
        print(sourceE,c)
        c+=1
        f.write(str(sourceE))
        f.write('\n')

def getMetric2(filename,n,K=2000,Norm=200):
    '''
    get metric from file for back to back link delay,need to calculate covariance
    :param filename:
    :param n: leaf nodes
    :param K: the sum back-to-back packet pairs received in pair leaf nodes
    :param Norm: the
    :return: a covariance needs Norm path delays to calculate
    '''
    SIJ = []
    f = open(filename, "r")
    while True:
        line = f.readline()
        if line:
            sij = ast.literal_eval(line)
            if len(sij) == K:
                if K == Norm:
                    ## 全部求均值
                    pathDelay1 = []
                    pathDelay2 = []
                    for item in sij:
                        pathDelay1.append(item[0])
                        pathDelay2.append(item[1])
                    cov = Covariance_way2(pathDelay1,pathDelay2)
                    SIJ.append(cov)
                else:
                    ##求度均值
                    temp = []
                    for i in range(int(K / Norm)):
                        pathDelay1 = []
                        pathDelay2 = []
                        for item in sij[i * Norm:(i + 1) * Norm]:
                            pathDelay1.append(item[0])
                            pathDelay2.append(item[1])
                        cov = Covariance_way2(pathDelay1, pathDelay2)
                        temp.append(cov)
                    SIJ.append(temp)
            else:
                print("getMetic中发现存在丢包")
                if K == Norm:
                    ## 全部求均值
                    pathDelay1 = []
                    pathDelay2 = []
                    for item in sij:
                        pathDelay1.append(item[0])
                        pathDelay2.append(item[1])
                    cov = Covariance_way2(pathDelay1, pathDelay2)
                    SIJ.append(cov)
                else:
                    ##求度均值
                    temp = []
                    partition = int(K / Norm)
                    norm = int(len(sij) / partition)
                    for i in range(partition):
                        pathDelay1 = []
                        pathDelay2 = []
                        if i + 1 == partition:
                            for item in sij[i * norm:]:
                                pathDelay1.append(item[0])
                                pathDelay2.append(item[1])
                            cov = Covariance_way2(pathDelay1, pathDelay2)
                            temp.append(cov)
                        else:
                            for item in sij[i * norm:(i + 1) * norm]:
                                pathDelay1.append(item[0])
                                pathDelay2.append(item[1])
                            cov = Covariance_way2(pathDelay1,pathDelay2)
                            temp.append(cov)
                    SIJ.append(temp)

        else:
            break
    if K == Norm:
        S = np.zeros((n, n))
        index = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    S[i][j] = S[j][i] = (SIJ[index] - 0.02)
                    index = index + 1
    else:
        S = {}
        index = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    key = "S" + str(i + 1) + "," + str(j + 1)
                    if index >= len(SIJ):
                        print("error")
                    for k in range(len(SIJ[index])):
                        SIJ[index][k] = SIJ[index][k] - 0.02
                    S[key] = SIJ[index]
                    index = index + 1
    return S


def getMetric3(filename,n,K=2000,Norm=200):
    '''
    for loss rate
    :return:
    '''
    SIJ = []
    f = open(filename, "r")
    while True:
        line = f.readline()
        if line:
            sij = ast.literal_eval(line)
            if len(sij) == K:
                if K == Norm:
                    ## 全部求均值
                    loss1 = 0 ##路径一的成功传输率
                    loss2 = 0 ##路径二的成功传输率
                    loss3 = 0 ##路径一和路径二全部成功传输率
                    for item in sij:
                        if item[2] == 1:
                            loss1+=1
                        if item[3] == 1:
                            loss2+=1
                        if item[2] == 1 and item[3] == 1:
                            loss3+=1
                    loss1 = loss1/2000
                    loss2 = loss2/2000
                    loss3 = loss3/2000
                    SIJ.append(loss1*loss2/loss3)
                else:
                    ##求度均值
                    temp = []
                    for i in range(int(K / Norm)):
                        loss1 = 0  ##路径一的成功传输率
                        loss2 = 0  ##路径二的成功传输率
                        loss3 = 0  ##路径一和路径二全部成功传输率
                        for item in sij[i * Norm:(i + 1) * Norm]:
                            if item[2] == 1:
                                loss1 += 1
                            if item[3] == 1:
                                loss2 += 1
                            if item[2] == 1 and item[3] == 1:
                                loss3 += 1
                        loss1 = loss1 / 200
                        loss2 = loss2 / 200
                        loss3 = loss3 / 200
                        temp.append(loss1*loss2/loss3)
                    SIJ.append(temp)

        else:
            break
    if K == Norm:
        S = np.zeros((n, n))
        index = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    S[i][j] = S[j][i] = (SIJ[index] - 0.02)
                    index = index + 1
    else:
        S = {}
        index = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    key = "S" + str(i + 1) + "," + str(j + 1)
                    if index >= len(SIJ):
                        print("error")
                    for k in range(len(SIJ[index])):
                        SIJ[index][k] = SIJ[index][k] - 0.02
                    S[key] = SIJ[index]
                    index = index + 1
    return S




if __name__ == "__main__":
    # S = calmetric2("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/sourceTrace0.tr","/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/order0",[4, 5, 5, 0, 4],2000)
    # calMetrics(calmetric2,saveS2,2000)
    # genSourceEs()
    doSim("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/light_load",getMetric,False)
    # pass
    # S = calmetric3("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/sourceTrace0.tr","/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/order0",[4, 5, 5, 0, 4],2000)
    # saveS2(S,"/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/result")
    # S = getMetric3("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/result", 3, K=2000, Norm=2000)
    # print(S)