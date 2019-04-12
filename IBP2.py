# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-3-26 上午10:57
@email:zongwang.zhang@outlook.com
'''


'''
改进代码 具有更高效率
先不考虑丢包
'''
from tool import *

def CalDelayCov(dir):
    '''
    计算路径时延，链路时延，计算路径协方差，计算链路方差，分别保存文件
    :return:
    '''
    sharePathNum = [8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        # 若存在文件则删除
        filename1 = data_dir + "/pathDelayCov" + str(sPN)
        filename2 = data_dir + "/linkDelayCov" + str(sPN)
        if os.path.exists(filename1):
            os.remove(filename1)
        if os.path.exists(filename2):
            os.remove(filename2)
        for cnt in range(10):
            filename = data_dir + "/" + "improved_b2b" + str(sPN) + "_" + str(cnt) + ".tr"
            print(filename)
            if os.path.exists(filename):
                filename3 = data_dir + "/" + "pathDelay" + str(sPN) + "_" + str(cnt)
                filename4 = data_dir + "/" + "linkDelay" + str(sPN) + "_" + str(cnt)
                pathDelayCov, linkDelayCov = calDelayCov(filename,sPN,filename3,filename4)
                print(pathDelayCov,np.sum(linkDelayCov))
                open(filename1,"a+").write(str(pathDelayCov))
                open(filename1, "a+").write("\n")
                open(filename2,"a+").write(str(linkDelayCov))
                open(filename2, "a+").write("\n")

def calDelayCov(filename,linknum,filename1,filename2):
    '''
    计算链路时延，路径时延，计算链路方差，路径协方差,使用默认的IP地址
    :param filename:
    :param linknum:
    :return:
    '''
    pathdelaycov = 0
    linkdelaycov = []
    if not os.path.exists(filename):
        print("file is not exist:",filename)
    else:
        NodeIn = [3+i for i in range(linknum)]
        ##初始化
        pathdelay = {} ##[packet_id,time,flag]
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
                key = str(i+2)+"-"+str(i+3)
                linkdelay["0-1"][key] = {}
                linkdelay["0-2"][key] = {}
        ##开始分析文件
        PATHDELAY = [[],[]] ##用来存储路径时延，此变量存入文件
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
                    if action == "r" and currentNode == 1 and src_ip == "10.1.1.1" and dest_ip == "10.1."+str(2+linknum)+".2" and dest_port == "9":
                        STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                        pathdelay["0-1"].append([packet_id,time-STARTTIME,True])
                        PATHDELAY[0].append(time-STARTTIME)
                    elif action == "r" and currentNode == 2 and src_ip == "10.1.1.1" and dest_ip == "10.1."+str(4+linknum)+".2" and dest_port == "10":
                        STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                        pathdelay["0-2"].append([packet_id,time-STARTTIME,True])
                        PATHDELAY[1].append(time - STARTTIME)

                    ##探测流的链路时延方差 路径1
                    if src_ip == "10.1.1.1" and dest_ip == "10.1."+str(2+linknum)+".2"  and currentNode != 1 and currentNode != 2 and dest_port != "8": ##说明是背景流 且在链路上
                        if action == "r" and currentNode != 3 and currentNode in NodeIn:
                            key = str(currentNode-1)+"-"+str(currentNode)
                            if packet_id not in linkdelay["0-1"][key]:
                                print("error")
                            else:
                                linkdelay["0-1"][key][packet_id] = time-linkdelay["0-1"][key][packet_id]
                            if currentNode+1 in NodeIn:
                                key = str(currentNode)+"-"+str(currentNode+1)
                                if packet_id in linkdelay["0-1"][key]:
                                    print("error")
                                else:
                                    linkdelay["0-1"][key][packet_id] = time
                        elif action == "r" and currentNode == 3:
                            key = "0-3"
                            STARTTIME = float(oc[51].split('=')[1])
                            linkdelay["0-1"][key][packet_id] = time-STARTTIME
                            if linknum != 1:
                                linkdelay["0-1"]["3-4"][packet_id] = time

                    if src_ip == "10.1.1.1" and dest_ip == "10.1." + str(4 + linknum) + ".2" and (
                            currentNode != 1 or currentNode != 2) and dest_port != 8:  ##说明是背景流 且在链路上
                        if action == "r" and currentNode != 3 and currentNode in NodeIn:
                            key = str(currentNode-1) + "-" + str(currentNode)
                            if packet_id not in linkdelay["0-2"][key]:
                                print("error")
                            else:
                                linkdelay["0-2"][key][packet_id] = time - linkdelay["0-2"][key][packet_id]
                            if currentNode+1 in NodeIn:
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



        ##存储文件

        #计算丢包，要么填充，要么删除无用包
        l_1 = len(pathdelay["0-1"])##路径1上接收包的数量
        l_2 = len(pathdelay["0-2"])##路径2上接收包的数量
        RTT = 0.3648*2
        if l_1 != 1000: ##说明在路径1上存在丢包
            lossrate1 = 1-(l_1/1000)
            print("路径1上丢包率为：",lossrate1)
            PATHDELAY[0].extend([RTT for i in range(1000-l_1)]) ##加上往返时延填充进去
        if l_2 != 1000:
            lossrate2 = 1-(l_2/1000)
            print("路径2上丢包率为：",lossrate2)
            PATHDELAY[1].extend([RTT for i in range(1000 - l_2)])##加上往返时延填充进去

        #计算协方差
        pathdelaycov = Covariance_way2(PATHDELAY[0],PATHDELAY[1])
        ##保存路径协方差
        np.savetxt(filename1,np.array(PATHDELAY))

        LINKDELAY = []##保存链路时延方差
        ##计算链路方差 暂时没考虑丢包
        for i in range(linknum):
            tempDelay = []
            if i == 0:
                key = "0-3"
            else:
                key = str(i+2)+"-"+str(i+3)
            for id in linkdelay["0-1"][key]:
                tempDelay.append(linkdelay["0-1"][key][id])


            for id in linkdelay["0-2"][key]:
                tempDelay.append(linkdelay["0-2"][key][id])
            LINKDELAY.append(tempDelay)
            linkdelaycov.append(Variance_way1(tempDelay))
        ##保存链路时延方差为文件
        np.savetxt(filename2,np.array(LINKDELAY))
    return pathdelaycov,linkdelaycov



def CalTrueLinkCov(dir):
    sharePathNum = [1,2,3,4,5,6,7,8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        # 若存在文件则删除
        filename1 = data_dir + "/TlinkDelayCov" + str(sPN)
        if os.path.exists(filename1):
            os.remove(filename1)
        for cnt in range(100):
            filename = data_dir + "/" + "improved_b2b" + str(sPN) + "_" + str(cnt) + ".tr"
            print(filename)
            if os.path.exists(filename):
                filename2 = data_dir + "/" + "TlinkDelay" + str(sPN) + "_" + str(cnt)
                linkDelayCov = calTrueLinkCov(filename, sPN, filename2)
                print(np.sum(linkDelayCov))
                open(filename1, "a+").write(str(linkDelayCov))
                open(filename1, "a+").write("\n")


def calTrueLinkCov(filename,linknum,filename1):
    '''
    计算链路时延方差(背景流)，暂时不考虑丢包，和在queuedisc上的排队时间
    背景流的抓取时间从探测流开始的时间10s开始
    :param filename:
    :return:
    '''
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


    DESTIP = []
    #源目的地址映射表
    matchTable = {}
    #初始化源目的映射表
    for i in range(linknum):
        destIP = "10.1." + str(i + 1) + ".2"
        DESTIP.append(destIP)
        if i == 0:
            matchTable[destIP] = "0-3"
        else:
            key = str(i + 2) + "-" + str(i + 3)
            matchTable[destIP]= key


    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                oc = re.split(r"\)|\(| ", line)
                action = oc[0]
                time = float(oc[1])
                packet_id = int(oc[23])
                dest_ip = oc[37]
                if oc[39] == 'ns3::TcpHeader':
                    dest_port = oc[43]
                elif oc[39] == 'ns3::UdpHeader':
                    dest_port = oc[45]
                #先确定是公共路径上的背景流
                if time >= 10 and time <= 20and dest_ip in DESTIP:
                    if action == "+":
                        key = matchTable[dest_ip]#查找key
                        if dest_port not in linkdelay[key]:
                            linkdelay[key][dest_port] = {}
                        linkdelay[key][dest_port][packet_id] = time
                    elif action == "r":
                        #因为是从探测流开始的时候来读取，所以部分包不存在开始的时间
                        key = matchTable[dest_ip]
                        if dest_port not in linkdelay[key]:
                            ##可以记录也可以丢弃，这里虽然记录了同时也丢弃了缺损的数据，包括丢包的数据和因为
                            ##截断时间的数据
                            linkdelay[key][dest_port] = {}
                            linkdelay[key][dest_port][packet_id] = time ##这里的时间应该会比较大
                        elif packet_id not in linkdelay[key][dest_port]:#说明这条流的其他包以及创建了dest_port
                            linkdelay[key][dest_port][packet_id] = time  ##这里的时间应该会比较大
                        else:
                            linkdelay[key][dest_port][packet_id] = time-linkdelay[key][dest_port][packet_id]
                            LINKDELAY[DESTIP.index(dest_ip)].append(linkdelay[key][dest_port][packet_id])

            else:
                break
    if os.path.exists(filename1):
        os.remove(filename1)
    linkdelaycov = []
    for item in LINKDELAY:
        linkdelaycov.append(Variance_way1(item))
        open(filename1,"a+").write(str(LINKDELAY))
        open(filename1,"a+").write("\n")
    return linkdelaycov


def plot_b2b(data_dir):
    '''
    plot end to end measurement of back to back probing
    :param data_dir:
    :return:
    '''
    PATHNUM = [1,2,3,4, 5, 6,7,8]

    plt.figure(1)
    curve1,curve2,curve3 = get_b2b(data_dir+"/b2b")
    curve4, curve5, curve6 = get_b2b(data_dir + "/ibp")
    fig = plt.subplot()
    plt.xlabel('#link on the shared path')
    plt.ylabel('estimated distance')
    plt.plot(PATHNUM, curve1, 'o-', label="Covariance (b2b)")
    plt.plot(PATHNUM, curve2, 'x-', label="sum variance(b2b)")
    plt.plot(PATHNUM, curve3, 'v-', label="true sum variance(b2b)")
    plt.plot(PATHNUM, curve4, '.-', label="Covariance (ibp)")
    plt.plot(PATHNUM, curve5, '<-', label="sum variance(ibp)")
    plt.plot(PATHNUM, curve6, '>-', label="true sum variance(ibp)")

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title("comparison(40%)")
    plt.show()
    plt.close()


    plt.figure(2)
    fig = plt.subplot(121)
    plt.plot(PATHNUM, np.array(curve3) - np.array(curve2), 'o-', label="absolute error(b2b)")
    plt.plot(PATHNUM, np.array(curve6) - np.array(curve5), 'o-', label="absolute error(ibp)")
    plt.legend()
    fig = plt.subplot(122)
    plt.plot(PATHNUM, (np.array(curve3) - np.array(curve2)) / np.array(curve2), 'x-', label="relative error(b2b)")
    plt.plot(PATHNUM, (np.array(curve6) - np.array(curve5)) / np.array(curve5), 'x-', label="relative error(ibp)")
    plt.legend()
    plt.show()
    plt.close()





def get_b2b(data_dir):
    '''
    得到背靠背包的cov(a,b),和sum_{X(a)}
    :return:
    '''
    curve1 = [] ##cov(a,b)
    curve2 = [] ##sum_{X(a)}
    curve3 = [] ##true sum_{X(a)}
    PATHNUM = [1,2,3,4, 5, 6, 7,8]
    for pathNum in PATHNUM:
        filename1 = data_dir+"/"+str(pathNum)+"/pathDelayCov"+str(pathNum)
        if os.path.exists(filename1):
            pathDelay = np.loadtxt(filename1)
            curve1.append(pathDelay.mean())
        filename2 = data_dir+"/"+str(pathNum)+"/linkDelayCov"+str(pathNum)
        if os.path.exists(filename2):
            linkDelay = []
            lines = open(filename2,'r').readlines()
            for line in lines:
                linkDelay.append(np.sum(ast.literal_eval(line)))
            curve2.append(np.mean(linkDelay))
        filename3 = data_dir+"/"+str(pathNum)+"/TlinkDelayCov"+str(pathNum)
        if os.path.exists(filename3):
            TlinkDelay = []
            lines = open(filename3,'r').readlines()
            for line in lines:
                TlinkDelay.append(np.sum(ast.literal_eval(line)))
            curve3.append(np.mean(TlinkDelay))

    return curve1,curve2,curve3

def plotPoint(filename1,filename2):
    fig = plt.subplot()
    lines = open(filename1,"r").readline()
    line = lines[1:-2]
    linkdelay = ast.literal_eval(line)
    for item in linkdelay:
        plt.scatter(1,item)
    plt.title("b2b")
    plt.ylim([0.018,0.022])
    plt.show()
    plt.close()

    fig = plt.subplot()
    lines = open(filename2, "r").readline()
    line = lines[1:-2]
    linkdelay = ast.literal_eval(line)
    for item in linkdelay:
        plt.scatter(1, item)
    plt.title("ibp")
    plt.ylim([0.018, 0.022])
    plt.show()
    plt.close()

def plotBVar(dir):
    '''
    画出背景流的方差曲线
    :param dir:
    :return:
    '''
    data_dir = dir+"/b2b"
    curve1 = []
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        filename1 = data_dir + "/" + str(pathNum) + "/TlinkDelayCov" + str(pathNum)
        if os.path.exists(filename1):
            TlinkDelay1 = []
            lines1 = open(filename1, 'r').readlines()
            for line in lines1:
                TlinkDelay1.append(np.sum(ast.literal_eval(line)))
            curve1.append(np.mean(TlinkDelay1))
    data_dir = dir+"/ibp"
    curve2 = []
    for pathNum in PATHNUM:
        filename2 = data_dir + "/" + str(pathNum) + "/TlinkDelayCov" + str(pathNum)
        if os.path.exists(filename2):
            TlinkDelay2 = []
            lines2 = open(filename2, 'r').readlines()
            for line in lines2:
                TlinkDelay2.append(np.sum(ast.literal_eval(line)))
            curve2.append(np.mean(TlinkDelay2))
    fig = plt.subplot()
    plt.xlabel('#link on the shared path')
    plt.ylabel('estimated distance')
    plt.plot(PATHNUM, curve1, 'o-', color = "black",label="true sum variance(b2b)")
    plt.plot(PATHNUM, curve2, 'x-', label="true sum variance(ibp)")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    # CalDelayCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data3/ibp")
    # CalTrueLinkCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/ibp")
    # pass
    # plot_b2b("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5")
    # plotPoint("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/b2b/1/TlinkDelay1_1","/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/ibp/1/TlinkDelay1_1")
    # plotBVar("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data5/test")
    filename1 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/test/1/back_to_back1_0.tr"
    filename2 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/test/1/pathDelay1_0"
    filename3 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data6/test/1/linkDelay1_0"
    calDelayCov(filename1,1,filename2,filename3)