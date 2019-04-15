# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-4-14 上午9:28
@email:zongwang.zhang@outlook.com


专门应对重载,也适用于
'''

from tool import *

def CalDelay(dir,probe_way):
    '''
        计算路径时延，链路时延,分别保存文件
        :return:
    '''
    sharePathNum = [3, 4, 5, 6, 7, 8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        for cnt in range(10):
            filename = data_dir + "/" + "back_to_back" + str(sPN) + "_" + str(cnt) + ".tr"
            print(filename)
            if os.path.exists(filename):
                filename1 = data_dir + "/" + "pathDelay" + str(sPN) + "_" + str(cnt)
                filename2 = data_dir + "/" + "linkDelay" + str(sPN) + "_" + str(cnt)
                calDelay(filename, sPN, filename1, filename2, probe_way)

def calDelay(filename,linknum,filename1,filename2,Probe_way="b2b"):
    pathdelaycov = 0
    linkdelaycov = []
    if not os.path.exists(filename):
        print("file is not exist:", filename)
    else:
        NodeIn = [3 + i for i in range(linknum)]
        ##初始化
        pathdelay = {}  ##[packet_id,time,flag]
        linkdelay = {}
        pathdelay["0-1"] = []
        pathdelay["0-2"] = []
        linkdelay["0-1"] = {}
        linkdelay["0-2"] = {}
        for i in range(linknum):
            if i == 0:
                linkdelay["0-1"]["0-3"] = {}
                linkdelay["0-2"]["0-3"] = {}
            else:
                key = str(i + 2) + "-" + str(i + 3)
                linkdelay["0-1"][key] = {}
                linkdelay["0-2"][key] = {}
        ##开始分析文件
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    oc = re.split(r"\)|\(| ", line)
                    action = oc[0]
                    time = float(oc[1])
                    namespace = oc[2]
                    packet_id = int(oc[23])
                    src_ip = oc[35]
                    dest_ip = oc[37]
                    dest_port = oc[45]
                    currentNode = int(namespace.split("/")[2])
                    ##路径时延，直接检查当前的节点
                    if action == "r" and currentNode == 1 and src_ip == "10.1.1.1" and dest_ip == "10.1." + str(
                            2 + linknum) + ".2" and dest_port == "9":
                        STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                        pathdelay["0-1"].append([packet_id, time - STARTTIME, True])
                    elif action == "r" and currentNode == 2 and src_ip == "10.1.1.1" and dest_ip == "10.1." + str(
                            4 + linknum) + ".2" and dest_port == "10":
                        STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                        pathdelay["0-2"].append([packet_id, time - STARTTIME, True])

                    ##探测流的链路时延方差 路径1
                    if src_ip == "10.1.1.1" and dest_ip == "10.1." + str(
                            2 + linknum) + ".2" and currentNode != 1 and currentNode != 2 and dest_port != "8":  ##说明是背景流 且在链路上
                        if action == "r" and currentNode != 3 and currentNode in NodeIn:
                            key = str(currentNode - 1) + "-" + str(currentNode)
                            if packet_id not in linkdelay["0-1"][key]:
                                print("error")
                            else:
                                linkdelay["0-1"][key][packet_id] = time - linkdelay["0-1"][key][packet_id]
                            if currentNode + 1 in NodeIn:
                                key = str(currentNode) + "-" + str(currentNode + 1)
                                if packet_id in linkdelay["0-1"][key]:
                                    print("error")
                                else:
                                    linkdelay["0-1"][key][packet_id] = time
                        elif action == "r" and currentNode == 3:
                            key = "0-3"
                            STARTTIME = float(oc[51].split('=')[1])
                            linkdelay["0-1"][key][packet_id] = time - STARTTIME
                            if linknum != 1:
                                linkdelay["0-1"]["3-4"][packet_id] = time

                    if src_ip == "10.1.1.1" and dest_ip == "10.1." + str(4 + linknum) + ".2" and (
                            currentNode != 1 or currentNode != 2) and dest_port != 8:  ##说明是背景流 且在链路上
                        if action == "r" and currentNode != 3 and currentNode in NodeIn:
                            key = str(currentNode - 1) + "-" + str(currentNode)
                            if packet_id not in linkdelay["0-2"][key]:
                                print("error")
                            else:
                                linkdelay["0-2"][key][packet_id] = time - linkdelay["0-2"][key][packet_id]
                            if currentNode + 1 in NodeIn:
                                key = str(currentNode) + "-" + str(currentNode + 1)
                                if packet_id in linkdelay["0-2"][key]:
                                    print("error")
                                else:
                                    linkdelay["0-2"][key][packet_id] = time
                        elif action == "r" and currentNode == 3:
                            key = "0-3"
                            STARTTIME = float(oc[51].split('=')[1])
                            linkdelay["0-2"][key][packet_id] = time - STARTTIME
                            if linknum != 1:
                                linkdelay["0-2"]["3-4"][packet_id] = time
                else:
                    break
        #计算丢包，并输出
        l_1 = len(pathdelay["0-1"])##路径1上接收包的数量
        l_2 = len(pathdelay["0-2"])##路径2上接收包的数量
        if l_1 != 1000: ##说明在路径1上存在丢包
            lossrate1 = 1-(l_1/1000)
            print("路径1上丢包率为：",lossrate1)
        if l_2 != 1000:
            lossrate2 = 1-(l_2/1000)
            print("路径2上丢包率为：",lossrate2)
        ##用来存储路径时延，此变量存入文件,丢失的包信息记录为1
        PATHDELAY = [
            [-1 for i in range(1000)],
            [-1 for i in range(1000)]
        ]
        for item1 in pathdelay["0-1"]:
            packetid = item1[0]
            if Probe_way == "b2b":
                PATHDELAY[0][packetid] = item1[1]
            elif Probe_way == "ibp":
                PATHDELAY[0][(packetid-1)/2] = item1[1]
        for item2 in pathdelay["0-2"]:
            packetid = item2[0]
            PATHDELAY[1][packetid] = item2[1]
        np.savetxt(filename1, np.array(PATHDELAY), fmt='%.10e')

        LINKDELAY = [[] for i in range(linknum)]  ##保存链路时延方差
        ##计算链路方差 暂时没考虑丢包
        for i in range(linknum):
            tempDelay = [
                [-1 for j in range(1000)],
                [-1 for j in range(1000)]
            ]
            if i == 0:
                key = "0-3"
            else:
                key = str(i + 2) + "-" + str(i + 3)
            for id in linkdelay["0-1"][key]:
                if Probe_way == "b2b":
                    tempDelay[0][id] = linkdelay["0-1"][key][id]
                elif Probe_way == "ibp":
                    tempDelay[0][(id-1)/2] = linkdelay["0-1"][key][id]
            for id in linkdelay["0-2"][key]:
                if Probe_way == "b2b":
                    tempDelay[1][id] = linkdelay["0-2"][key][id]
                elif Probe_way == "ibp":
                    tempDelay[1][(id-1)/2] = linkdelay["0-2"][key][id]
            # l = len(tempDelay[0])+len(tempDelay[1])  ##key这条链路上收集的探测流的包的数目
            # lossrate3 = 1 - (l / 2000)
            # print("loss rate in", key, lossrate3) ##因为记录文件的方式变了 所以丢包率输出方式不一样
            LINKDELAY[i].extend(tempDelay[0])
            LINKDELAY[i].extend(tempDelay[1])
        ##保存链路时延方差为文件
        np.savetxt(filename2, np.array(LINKDELAY))

def CalTrueLinkDelay(dir):
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        # 若存在文件则删除
        for cnt in range(100):
            filename = data_dir + "/" + "back_to_back" + str(sPN) + "_" + str(cnt) + ".tr"
            print(filename)
            if os.path.exists(filename):
                filename2 = data_dir + "/" + "TlinkDelay" + str(sPN) + "_" + str(cnt)
                calTrueLinkDelay(filename, sPN, filename2)

def calTrueLinkDelay(filename, linknum, filename1):
    '''
    计算链路时延(背景流)
    背景流的抓取时间从探测流开始的时间10s开始
    :param
    filename:
    :return:
    '''
    ##记录丢包信息
    lossInfo = [[0,0,0] for i in range(linknum)] ##before after loss
    #保存这一条链路上的所有的时延
    LINKDELAY = [[] for i in range(linknum)]
    #记录各个流的时延
    linkdelay = {}
    #初始化linkdelay
    for i in range(linknum):
        if i == 0:
            linkdelay["0-3"] = {}
        else:
            key = str(i + 2) + "-" + str(i + 3)
            linkdelay[key] = {}
    #keyset
    keyset = []
    for i in range(linknum):
        if i == 0:
            keyset.append("0-3")
        else:
            key = str(i + 2) + "-" + str(i + 3)
            keyset.append(key)

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                oc = re.split(r"\)|\(| ", line)
                action = oc[0]
                namespace = oc[2]
                time = float(oc[1])
                packet_id = int(oc[23])
                dest_ip = oc[37]
                if oc[39] == 'ns3::TcpHeader':
                    dest_port = oc[43]
                elif oc[39] == 'ns3::UdpHeader':
                    dest_port = oc[45]
                currentNode = int(namespace.split("/")[2])
                #先确定是公共路径上的背景流
                if action =="r" and time >= 10 and time <= 21 and matchIP(dest_ip,linknum)[0]: ##这里设置的time时间条件的原因是要处理一个边界问题.
                    key = matchIP(dest_ip,linknum)[1]#查找key
                    if currentNode == int(re.split(r"\-",key)[0]):
                        if time <= 20:
                            if dest_port not in linkdelay[key]:##说明这个包是这条流的"第一个"报文
                                linkdelay[key][dest_port] = {}
                                linkdelay[key][dest_port][packet_id] = time
                            else:
                                linkdelay[key][dest_port][packet_id] = time
                            lossInfo[keyset.index(key)][0]+=1
                    elif currentNode == int(re.split(r"\-",key)[1]):
                        #因为是从探测流开始的时候来读取，所以部分包不存在开始的时间
                        key = matchIP(dest_ip, linknum)[1]  # 查找key
                        if dest_port not in linkdelay[key]:
                            ##这个数据还是因为截取时间而导致无用的数据
                            continue
                        elif packet_id not in linkdelay[key][dest_port]:
                            #起始时间大于20的包文
                            ##或者该包的起始时间在10之前,但是被后面的包记录了端口,所以出现在这个条件下
                            continue
                        else:
                            linkdelay[key][dest_port][packet_id] = time-linkdelay[key][dest_port][packet_id]
                            LINKDELAY[keyset.index(key)].append(linkdelay[key][dest_port][packet_id])
                            lossInfo[keyset.index(key)][1] += 1

            else:
                break

    for i in range(len(keyset)):
        losslen = lossInfo[i][2]
        LINKDELAY[i].extend([-1 for j in range(losslen)])
    if os.path.exists(filename1):
        os.remove(filename1)
    for item in LINKDELAY:
        open(filename1,"a+").write(str(item))
        open(filename1,"a+").write("\n")
    #保存了丢包信息.可以选择输出

def matchIP(ip,linknum):
    maxNode = linknum+2 ##公共路径上最大的节点
    segIP = re.split(r"\.",ip)
    destNode = int(segIP[2])
    if segIP[0] == "10" or segIP[0] == "8" or segIP[0] == "9" or destNode > maxNode or destNode == 1 or destNode == 2:
        return False,0
    else:
        if destNode == 3:
            key = "0-3"
        elif destNode <= maxNode:
            key = str(destNode-1)+"-"+str(destNode)
        return True,key
def CalCov(dir):
    '''
    分别计算pathdelaycov,linkdelaycov,Tlinkdelaycov,两种计算方式
    :param dir:
    :return:
    '''
    sharePathNum = [3, 4, 5, 6, 7, 8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        # 若存在文件则删除
        calCov(data_dir,sPN,"discard")


def calCov(dir,linknum,way):
    '''
    -1代表丢失的数据,两种方式计算
    :param dir:
    :return:
    '''
    RTT = 0.36
    ##计算linkdelaycov
    linkdelayCov = []
    filename1 = dir+"/linkDelayCov"+str(linknum)
    if os.path.exists(filename1):
        os.remove(filename1)
    for c in range(100):
        linkdelaycov = []
        f1 = dir+"/linkDelay"+str(linknum)+"_"+str(c)
        if not os.path.exists(f1):
            print(f1,"not exist")
            continue
        linkdelays = np.loadtxt(f1)
        if len(linkdelays) >8:
            linkdelays = [linkdelays]
        for linkdelay in linkdelays:
            if way == "discard":
                LinkDelay = []
                linkdelay1 = linkdelay[0:1000]
                linkdelay2 = linkdelay[1000:2000]
                for i in range(len(linkdelay1)):
                    if linkdelay1[i] >0 and linkdelay2[i] > 0:
                        LinkDelay.append(linkdelay1[i])
                        LinkDelay.append(linkdelay2[i])
                cov = Variance_way1(LinkDelay)
            elif way == "use":
                for i in range(len(linkdelay)):
                    if linkdelay[i]<0:
                        linkdelay[i] = RTT
                cov = Variance_way1(linkdelay)
            linkdelaycov.append(cov)
        linkdelayCov.append(linkdelaycov)
    ##计算pathdelaycov
    pathDelayCov = []
    filename2 = dir+"/pathDelayCov"+str(linknum)
    if os.path.exists(filename2):
        os.remove(filename2)
    for c in range(100):
        f2 = dir+"/pathDelay"+str(linknum)+"_"+str(c)
        if not os.path.exists(f2):
            print(f2,"not exist")
            continue
        pathdelay = np.loadtxt(f2)
        if way == "discard":
            PathDelay = [[],[]]
            for i in range(len(pathdelay[0])):
                if pathdelay[0][i] > 0 and pathdelay[1][i] > 0:
                    PathDelay[0].append(pathdelay[0][i])
                    PathDelay[1].append(pathdelay[1][i])
            cov = Covariance_way2(PathDelay[0],PathDelay[1])
        elif way == "use":
            for i in range(len(pathdelay[0])):
                if pathdelay[0][i]<0:
                    pathdelay[0][i] = RTT*(linknum+2)
                if pathdelay[1][i]<0:
                    pathdelay[1][i] = RTT*(linknum+2)
            cov = Covariance_way2(pathdelay[0],pathdelay[1])
        pathDelayCov.append(cov)

    ##计算TlinkdelayCov
    TlinkdelayCov = []
    filename3 = dir+"/TlinkDelayCov"+str(linknum)
    if os.path.exists(filename3):
        os.remove(filename3)
    for c in range(100):
        Tlinkdelaycov = []
        f3 = dir+"/TlinkDelay"+str(linknum)+"_"+str(c)
        if not os.path.exists(f3):
            print(f3,"not exist")
            continue
        lines = open(f3,"r").readlines()
        for line in lines:
            Tlinkdelay = ast.literal_eval(line)
            if way == "discard":
                TLinkDelay = []
                for i in range(len(Tlinkdelay)):
                    if Tlinkdelay[i] > 0:
                        TLinkDelay.append(Tlinkdelay[i])
                cov = Variance_way1(TLinkDelay)

            elif way == "use":
                for i in range(len(Tlinkdelay)):
                    if Tlinkdelay[i] < 0:
                        Tlinkdelay[i] = RTT
                cov = Variance_way1(Tlinkdelay)
            Tlinkdelaycov.append(cov)
        TlinkdelayCov.append(Tlinkdelaycov)
    np.savetxt(filename1,linkdelayCov)
    np.savetxt(filename2,pathDelayCov)
    np.savetxt(filename3,TlinkdelayCov)



if __name__ == "__main__":
  # CalDelay("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b",probe_way="b2b")
  # CalTrueLinkDelay("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b")
  # CalCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b")
  # CalTrueLinkCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/ibp")
  # pass
  # plot_b2b("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5")
  # plotPoint("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/b2b/1/TlinkDelay1_1","/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/ibp/1/TlinkDelay1_1")
  # plotBVar("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/test")
  # filename1 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b/1/back_to_back1_0.tr"
  # filename2 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b/1/pathDelay1_0"
  # filename3 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/b2b/1/linkDelay1_0"
  # calDelayCov(filename1,1,filename2,filename3)
  pass