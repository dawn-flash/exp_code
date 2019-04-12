'''
轻载同构三明治探测分析代码 大机器上用
'''
import os
import time
import re
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

def calMetrics(dir,pktnum=200):
    '''
    计算度均值
    :param pktnum:
    :return:
    '''
    VTrees = getVTrees(dir+"/Topo_4_3_10")
    n = len(VTrees)
    while len(VTrees) != 0:
        ##还有文件没处理完
        flag = False ##表示没有文件要处理
        f1 = " "
        f2 = " "
        for i in range(n):
            filename1 = dir+"/sourceTrace" + str(
                i) + ".tr"
            filename2 = dir+"/Metric" + str(i)
            filename3 = dir+"/sourceTrace" + str(
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
            # os.remove(f1)
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
                                            if key == str(1)+","+str(3):
                                                print(key)
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

if __name__ == "__main__":
    dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/sandwich_vary_all"
    calMetrics(dir)