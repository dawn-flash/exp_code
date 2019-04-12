# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-3-19 下午6:45
@email:zongwang.zhang@outlook.com
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
                if src_ip == srcIP and dest_ip == destIp and currentNode == destnode and action == "r":
                    SeqTsHeader = oc[47]
                    STARTTIME = float(oc[51].split('=')[1])  ## 程序的发包时间
                    delay.append(time-STARTTIME)
                    rec_id.append(packet_id)
            else:
                break

    data = {
        "mean_delay": np.mean(delay),
        "loss rate": 1-(len(rec_id) / 1000),
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
                if src_ip == srcIP and dest_ip in destIPList:  ## 为探测包
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


def calDelayCov(dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data3/end_to_end/back-to-back2/",load="medium_load"):
    '''
    计算路径时延，链路时延，计算路径协方差，计算链路方差，分别保存文件
    :return:
    '''
    sharePathNum = [1,2,3,4,5,6,7,8]
    for sPN in sharePathNum:
        data_dir = dir + "/"+load+"/" + str(sPN)
        #若存在文件则删除
        filename1 = data_dir + "/pathDelayCov" + str(sPN)
        filename2 = data_dir + "/linkDelayCov" + str(sPN)
        if os.path.exists(filename1):
            os.remove(filename1)
        if os.path.exists(filename2):
            os.remove(filename2)
        for cnt in range(100):
            filename = data_dir + "/" + load + str(sPN) + "_" + str(cnt) + ".tr"
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
                if len(pathDelay1) == 1000 and len(pathDelay2) == 1000:
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
    # LOAD = ["light_load", "medium_load", "heavy_load"]
    LOAD = ["medium_load","heavy_load"]
    # LABEL = ["9%|0", "40%|0", "75%|1%"]
    LABEL = ["40%|0","75%|1%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    # PATHNUM = [1]
    for load in LOAD:
        curve1,curve2,std1,std2,se1,se2 = get_b2b(data_dir, load)
        fig = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')

        # plt.ylim([0, 3e-7])

        label = LABEL[LOAD.index(load)]
        # plt.plot(PATHNUM, curve1, 'o-', label="Covariance")
        # plt.plot(PATHNUM,curve2,'x-',label="Sum Variance")
        plt.errorbar(PATHNUM, curve1, std1, fmt="-o", )
        plt.errorbar(PATHNUM, curve2, std2, fmt="-.")
        plt.errorbar(PATHNUM, curve1, std1, fmt="-o", label="Covariance")
        plt.errorbar(PATHNUM, curve2, std2, fmt="-.", label="Sum Variance")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title(label+"     b2b")
        print("绝对误差：", np.array(curve1) - np.array(curve2))
        print("协方差：", curve1)
        print("方差和：", curve2)
        print("协方差的标准差:", std1)
        print("方差和的标准差：", std2)
        print("协方差的标准误差:", se1)
        print("方差和的标准误差：", se2)

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
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    # PATHNUM = [1]
    for pathNum in PATHNUM:
        filename1 = data_dir+"/"+load+"/"+str(pathNum)+"/pathDelayCov"+str(pathNum)
        if os.path.exists(filename1):
            pathDelay = np.loadtxt(filename1)
            std1.append(np.std(pathDelay))
            curve1.append(pathDelay.mean())
            se1.append(np.std(pathDelay) / np.sqrt(len(pathDelay)))
        filename2 = data_dir+"/"+load+"/"+str(pathNum)+"/linkDelayCov"+str(pathNum)
        if os.path.exists(filename2):
            linkDelay = []
            lines = open(filename2,'r').readlines()
            for line in lines:
                linkDelay.append(np.sum(ast.literal_eval(line)))
            curve2.append(np.mean(linkDelay))
            std2.append(np.std(linkDelay))
            se2.append(np.std(linkDelay) / len(linkDelay))
    return curve1,curve2,std1,std2,se1,se2
def plot_error_b2b(data_dir):
    '''
    画方差的误差比例
    :return:
    '''
    LOAD = ["medium_load"]
    LABEL = ["40%|0"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7,8]
    for load in LOAD:
        curve1, curve2 = get_b2b(data_dir, load) ## cov sum
        curve1 = np.array(curve1)
        curve2 = np.array(curve2)
        curve3 = curve2-curve1
        curve3 = curve3/curve2
        fig1 = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('error proportion')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve3, 'o-', label=label + "(cov-sum_var)/sum_var", c="blue")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title('b2b Probing(delay cov)', loc='center')
        plt.show()
if __name__ == "__main__":
    calDelayCov("/media/zongwangz/RealPAN-13438811621/myUbuntu/data4/end-to-end/back-to-back","heavy_load")
    # plot_b2b("/media/zongwangz/RealPAN-13438811621/myUbuntu/data4/end-to-end/back-to-back")
    # plot_error_b2b("/media/zongwangz/RealPAN-13438811621/myUbuntu/data4/end-to-end/back-to-back")