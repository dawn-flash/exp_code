# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-3-27 上午8:30
@email:zongwang.zhang@outlook.com
'''
'''
增大发包数量
'''
from tool import *
def getPathDelay(srcIP, destIp, filename, destnode):
    '''
    从一个trace文件中得到一条路径上的所有包的delay，记录得到的包id
    :param srcIP:
    :param destIp:
    :param filename:
    :param destnode:
    :return: 返回丢包率，平均路径时延，路径时延
    '''
    delay = []
    rec_id = []
    with open(filename,'r') as f:
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
                src_port = oc[43]
                dest_port = oc[45]
                currentNode = int(namespace.split("/")[2])
                if src_ip == srcIP and dest_ip == destIp and currentNode == destnode and action == "r" and dest_port != "8":
                    SeqTsHeader = oc[47]
                    STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                    delay.append(time-STARTTIME)
                    rec_id.append(packet_id)
            else:
                break

    data = {
        "mean_delay": np.mean(delay),
        "loss rate": 1-(len(rec_id) / 5000),
        "delay": delay,
        "rec_id":rec_id, ##丢包id 可以从中取得
    }
    return data


def getLinkDelay(srcIP, destIPList, filename, headNode, tailNode):
    '''
    获取一条链路上背靠背包在这条链路上的时延，丢包
    时延计算为从上一个节点 r 到下一个时间 r
    源节点则是程序的发包时间
    还未测试
    :return: 丢包率，时延
    '''

    record = {} ##记录这条链路上包id和时间
    record[1] = {}
    record[1]["before"] = 0
    record[1]["after"] = 0
    record[2] = {}
    record[2]["before"] = 0
    record[2]["after"] = 0
    linkDelay = [] ##记录返回的链路时延
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
                src_port = oc[43]
                dest_port = oc[45]
                currentNode = int(namespace.split("/")[2])
                if src_ip == srcIP and dest_ip in destIPList and dest_port != "8":  ## 为探测包
                    SeqTsHeader = oc[47]
                    STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                    if currentNode == headNode: ##在头结点处
                        if headNode == 0: ##头结点为源节点，则记录发包时间
                            key = destIPList.index(dest_ip)+1
                            if packet_id not in record[key]:
                                record[key][packet_id] = STARTTIME
                                record[key]["before"] += 1
                        elif action == "r": ##头结点为非源节点
                            key = destIPList.index(dest_ip)+1
                            if packet_id not in record[key]:
                                record[key][packet_id] = time
                                record[key]["before"] += 1
                            else:  ## 有错误产生 被记录过多次
                                print("error !")
                    if currentNode == tailNode and action == "r":
                        key = destIPList.index(dest_ip)+1
                        if packet_id in record[key]:
                            record[key][packet_id] = time - record[key][packet_id]
                            record[key]["after"] += 1
                            linkDelay.append(record[key][packet_id])
                        else:
                            print(packet_id,"to node",str(key),"lost in link",str(headNode),str(tailNode))
            else:
                break
    loss_rate = 1-((record[1]["after"]+record[2]["after"])/(record[1]["before"]+record[2]["before"]))
    data = {
        "loss_rate":loss_rate,
        "linkDelay":linkDelay,
    }
    return data


def calDelayCov(dir):
    '''
    计算路径时延，链路时延，计算路径协方差，计算链路方差，分别保存文件
    :return:
    '''
    sharePathNum = [8]
    for sPN in sharePathNum:
        data_dir = dir + "/" + str(sPN)
        #若存在文件则删除
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
                data1 = getPathDelay("10.1.1.1", "10.1." + str(2 + sPN) + ".2", filename, 1)
                data2 = getPathDelay("10.1.1.1", "10.1." + str(4 + sPN) + ".2", filename, 2)
                pathDelay1 = data1["delay"]
                pathDelay2 = data2["delay"]
                rec_id1 = data1["rec_id"]
                rec_id2 = data2["rec_id"]
                # 注：pathDelay中的顺序对应着rec_id
                hatLinkDelay1 = [] ##记录所有包时延，丢失时间按2倍的往返时延计算
                hatLinkDelay2 = []
                if len(pathDelay1) == 5000 and len(pathDelay2) == 5000:
                    hatLinkDelay1 = pathDelay1
                    hatLinkDelay2 = pathDelay2
                else:
                    for i in range(1000):
                        if i in rec_id1:
                            hatLinkDelay1.append(pathDelay1[rec_id1.index(i)])
                        if i not in rec_id1:
                            hatLinkDelay1.append(2*0.3648*(sPN+2))
                        if i in rec_id2:
                            hatLinkDelay2.append(pathDelay2[rec_id2.index(i)])
                        if i not in rec_id2:
                            hatLinkDelay2.append(2*0.3648*(sPN+2))
                pathDelayCov = Covariance_way2(hatLinkDelay1, hatLinkDelay2)
                pathDelay = [hatLinkDelay1,hatLinkDelay2]
                open(filename1,"a+").write(str(pathDelayCov)+'\n')
                filename3 = data_dir + "/pathDelay" + str(sPN) + "_" + str(cnt)
                np.savetxt(filename3,pathDelay)
                LinkDelay = []
                for i in range(sPN):
                    if i == 0:
                        LinkDelay.append(getLinkDelay("10.1.1.1", ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"], filename, 0, 3)["linkDelay"])
                        LinkDelay[-1].extend([2*0.3648 for i in range(2000-len(LinkDelay[-1]))])
                    else:
                        LinkDelay.append(getLinkDelay("10.1.1.1", ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"], filename, i+2, i+3)["linkDelay"])
                        LinkDelay[-1].extend([2 * 0.3648 for i in range(2000 - len(LinkDelay[-1]))])
                linkDelayCov = []
                for linkdelay in LinkDelay:
                    linkDelayCov.append(Covariance_way2(linkdelay,linkdelay))
                filename4 = data_dir + "/linkDelay" + str(sPN) + "_" + str(cnt)
                open(filename4,"w").write(str(LinkDelay)+"\n")
                open(filename2,"a+").write(str(linkDelayCov)+"\n")
                print(pathDelayCov,sum(linkDelayCov))
def plot_b2b(data_dir):
    '''
    plot end to end measurement of back to back probing
    :param data_dir:
    :return:
    '''
    LOAD = ["medium_load"]
    LABEL = ["40%|0"]
    PATHNUM = [1,2,3,4, 5, 6, 7,8]
    for load in LOAD:
        plt.figure(1)
        curve1,curve2,std1,std2,se1,se2 = get_b2b(data_dir, load)
        fig = plt.subplot()
        plt.xlabel('#link on the shared path')
        plt.ylabel('estimated distance')
        plt.ylim([0,5e-6])
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label="Covariance")
        plt.plot(PATHNUM, curve2, 'x-', label="sum variance")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title(label + "    ibp")
        plt.show()
        plt.close()


        plt.figure(2)
        fig = plt.subplot(121)
        print("绝对误差：", np.array(curve1) - np.array(curve2))
        plt.plot(PATHNUM, np.array(curve1) - np.array(curve2), 'o-', label="absolute error")
        plt.legend()
        fig = plt.subplot(122)
        plt.plot(PATHNUM, (np.array(curve1) - np.array(curve2)) / np.array(curve2), 'x-', label="relative error ")
        plt.legend()
        plt.show()
        plt.close()





def get_b2b(data_dir, load):
    '''
    得到背靠背包的cov(a,b),和sum_{X(a)}
    :return:
    '''
    curve1 = [] ##cov(a,b)
    std1 = []
    curve2 = [] ##sum_{X(a)}
    std2 =[]
    se1 = []
    se2 = []
    PATHNUM = [1,2,3,4, 5, 6, 7,8]
    # PATHNUM = [1]
    for pathNum in PATHNUM:
        filename1 = data_dir+"/"+str(pathNum)+"/pathDelayCov"+str(pathNum)
        if os.path.exists(filename1):
            pathDelay = np.loadtxt(filename1)
            std1.append(np.std(pathDelay))
            curve1.append(pathDelay.mean())
            se1.append(np.std(pathDelay)/np.sqrt(len(pathDelay)))
        filename2 = data_dir+"/"+str(pathNum)+"/linkDelayCov"+str(pathNum)
        if os.path.exists(filename2):
            linkDelay = []
            lines = open(filename2,'r').readlines()
            for line in lines:
                linkDelay.append(np.sum(ast.literal_eval(line)))
            curve2.append(np.mean(linkDelay))
            std2.append(np.std(linkDelay))
            se2.append(np.std(linkDelay)/len(linkDelay))
    return curve1,curve2,std1,std2,se1,se2


def plot_some():
    '''
    随便画点什么
    :return:
    '''
    pathNum = [4,5,6,7,8]
    curve1 = [1.0934839165124402e-07, 1.2711599147281793e-07, 1.5119791427911669e-07, 1.8356163234601812e-07, 2.1305472895987242e-07]
    curve2 = [1.0279050012648515e-07, 1.1359626278668627e-07, 1.3930471161005952e-07, 1.6777519687105875e-07, 1.7827273777535635e-07]
    curve3 = [1.1914729631587808e-07, 1.636954946825744e-07, 1.905306374606404e-07, 2.2838120238940808e-07, 2.599925859012901e-07]
    curve4 = [9.681261541894966e-08, 1.3036048298994933e-07, 1.381572066032417e-07, 1.7433216229436497e-07, 1.956772274564775e-07]
    fig = plt.subplot()
    plt.xticks([4,5,6,7,8])
    plt.xlabel('sharedLinkNum')
    plt.ylabel('sharedPathDistance')
    plt.plot(pathNum, curve1, 'o-', label="Sum Variance(b2b)")
    plt.plot(pathNum, curve2, 'x-', label="covariance(b2b)")
    plt.plot(pathNum, curve3, '.-', label="Sum Variance(improved b2b)")
    plt.plot(pathNum, curve4, 'v-', label="covariance(improved b2b)")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title("40%|0%")
    plt.show()
    plt.close()

    fig = plt.subplot()
    plt.xticks([4, 5, 6, 7, 8])
    plt.xlabel('sharedLinkNum')
    plt.ylabel('sharedPathDistance')
    plt.plot(pathNum, [2.23346809e-08 ,3.33350117e-08,5.23734309e-08 ,5.40490401e-08,6.43153584e-08], 'o-', label="绝对误差(improved b2b)")
    plt.plot(pathNum, [6.55789152e-09,1.35197287e-08 ,1.18932027e-08,1.57864355e-08, 3.47819912e-08], 'x-', label="绝对误差(b2b)")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title("40%|0%")
    plt.show()
    plt.close()


def effect1(filename):
    '''
    检查两小包到达分支节点时间
    :param filename:
    :return:
    '''
    time1 = 0
    time2 = 0
    cnt = 0
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
                src_port = oc[43]
                dest_port = oc[45]
                currentNode = int(namespace.split("/")[2])
                if currentNode == 6 and action == "r":
                    if time1 == 0:
                        time1 = time
                    elif time1 != 0 and time2 == 0:
                        time2 = time
                    elif time1 != 0 and time2 != 0:
                        if (time-time2)>0.00003:
                            print(packet_id,time-time2)
                            print("false")
                            cnt += 1
                        time1 = 0
                        time2 = 0

                    #
                    # if time1 == 0:
                    #     time1 = time
                    # else:
                    #     print(packet_id,time-time1)
                    #     print(packet_id, time - time1)
                    #     if (time - time1) > 0.00003:
                    #         print("false")
                    #         cnt += 1
                    #     time1 = 0
            else:
                break
    print(cnt)


def effect2(filename):
    '''
    检查两小包到达分支节点时间
    :param filename:
    :return:
    '''
    time1 = 0
    # time2 = 0
    cnt = 0
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
                src_port = oc[43]
                dest_port = oc[45]
                currentNode = int(namespace.split("/")[2])
                if currentNode == 7 and action == "r":
                    # if time1 == 0:
                    #     time1 = time
                    # elif time1 != 0 and time2 == 0:
                    #     time2 = time
                    # elif time1 != 0 and time2 != 0:
                    #     print(packet_id,time-time2)
                    #     if (time-time2)>0.00003:
                    #         print("false")
                    #         cnt += 1
                    #     time1 = 0
                    #     time2 = 0


                    if time1 == 0:
                        time1 = time
                    else:
                        if (time - time1) > 0.00005:
                            print("false")
                            print(packet_id, time - time1)
                            cnt += 1
                        time1 = 0
            else:
                break
    print(cnt)


def plot_b2bWithLog(data_dir):
    '''
    画图，使用log形式的时延
    :param data_dir:
    :return:
    '''
    LOAD = ["medium_load"]
    LABEL = ["40%|0"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7]
    for load in LOAD:
        plt.figure(1)
        curve1, curve2 = get_b2bWithLog(data_dir)
        fig = plt.subplot()
        plt.xlabel('#link on the shared path')
        plt.ylabel('estimated distance')
        # plt.ylim([0,3e-7])
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label="Covariance")
        plt.plot(PATHNUM, curve2, 'x-', label="sum variance")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title(label + "   b2b")
        plt.show()
        plt.close()

        plt.figure(2)
        fig = plt.subplot(121)
        print("绝对误差：", np.array(curve1) - np.array(curve2))
        plt.plot(PATHNUM,np.array(curve1) - np.array(curve2),'o-', label="absolute error")
        fig = plt.subplot(122)
        plt.plot(PATHNUM,(np.array(curve1) - np.array(curve2))/np.array(curve2),'x-', label="relative error ")


        plt.show()
        plt.close()


def get_b2bWithLog(data_dir):
    '''
    从基本的文件中使用对数时延计算协方差
    :param data_dir:
    :param load:
    :return:
    '''
    curve1 = []  ##cov(a,b)
    curve2 = []  ##sum_{X(a)}
    PATHNUM = [1, 2, 3, 4, 5, 6, 7]
    for pathNum in PATHNUM:
        pathDelayCov = []
        for i in range(100):
            filename1 = data_dir + "/" + str(pathNum) + "/pathDelay" + str(pathNum) + "_" + str(i)
            pathDelay = np.loadtxt(filename1)
            pathDelay = np.log10(pathDelay)  ##log化
            pathDelayCov.append(Covariance_way2(pathDelay[0], pathDelay[1]))

        curve1.append(np.mean(pathDelayCov))
        linkDelayCov = []
        for i in range(100):
            filename2 = data_dir + "/" + str(pathNum) + "/linkDelay" + str(pathNum) + "_" + str(i)
            lines = open(filename2).readlines()
            linkDelay = ast.literal_eval(lines[0])
            cov = 0
            for j in range(len(linkDelay)):
                cov += Variance_way1(np.log10(linkDelay[j]))
            linkDelayCov.append(cov)
        curve2.append(np.mean(linkDelayCov))
    return curve1, curve2

if __name__ == "__main__":
    calDelayCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data3/ibp")
    # plot_b2b("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data2/ibp")
    # plot_some()
    # effect2("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data2/b2b/5/back_to_back5_0.tr")
    # effect2("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/back-to-back/medium_load/3/medium_load3_1.tr")
    # effect2("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data2/b2b/5/back_to_back5_0.tr")
    # effect2("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data3/b2b/5/back_to_back5_0.tr")
    # pass
    # plot_b2bWithLog("/media/zongwangz/RealPAN-13438811621/myUbuntu/b2b_improvement/data2/b2b")