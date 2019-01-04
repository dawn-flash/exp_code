import os
import re
import time
import sys
'''
三种重载loss分析代码 大机器上用
'''
data_dir = "/home/zzw/data/heavy_load"
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

def getVTrees(filename="/home/zzw/data/light_load/Topo_4_3_10"):
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


def calMetrics(calmetric,saveS,pktnum=200):
    '''
    to calculate the average metric
    :param: calmetric   the function to calculate metrics
    :param pktnum:
    :return:
    '''
    VTrees = getVTrees(data_dir+"/Topo_4_3_10")
    n = len(VTrees)
    while len(VTrees) != 0:
        ##还有文件没处理完
        flag = False ##表示没有文件要处理
        f1 = " "
        f2 = " "
        for i in range(n):
            # if i >= 100:
            #     print("for medium b2b 0 done")
            #     sys.exit(-1)
            filename1 = data_dir+"/sourceTrace" + str(i) + ".tr"
            filename2 = data_dir+"/Metric" + str(i)
            filename3 = data_dir+"/sourceTrace" + str(i+1) + ".tr"
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
            f3 = data_dir+"/order"+str(i)
            S = calmetric(f1,f3,VTrees[i])
            saveS(S,f2)
            os.remove(f1)
            os.remove(f3)


if __name__ == "__main__":
    calMetrics(calmetric3,saveS2,2000)