'''
用来处理ns3的trace文件
'''
import re
import numpy as np
from matplotlib import pyplot as plt
import os
import copy
from scipy.stats import norm
from matplotlib import mlab
from scipy.stats import kstest
from scipy.stats import lognorm
import scipy.stats as stats
def getPathDelay(srcIP, destIp,filename,destnode):
    '''
    从一个trace文件中得到一条路径上的所有包的delay，记录得到的包id
    :param srcIP:
    :param destIp:
    :param filename:
    :param destnode:
    :return: 返回丢包率，平均路径时延，路径时延
    '''
    highest_packet_id = 0
    start_time = {}
    end_time = {}
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ",line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            src_port = oc[43]
            dest_port = oc[45]
            if src_ip == srcIP and dest_ip == destIp:
                if packet_id > highest_packet_id:
                    highest_packet_id = packet_id
                if packet_id not in start_time:
                    start_time[packet_id] = time
                if action == 'r':
                    flag = namespace.find('NodeList/'+str(destnode))
                    if flag == -1:
                        end_time[packet_id] = -1
                    else:
                        end_time[packet_id] = time
    rec_id = []
    delay = []
    delaySum = 0
    cnt = 0
    for id in range(highest_packet_id+1):
        if id in start_time and id in end_time:
            packet_duration = end_time[id]-start_time[id] ##一个包的延迟
            rec_id.append(id)
            if start_time[id] < end_time[id]:
                delaySum = packet_duration+delaySum ##所有包的延迟
                delay.append(packet_duration)
                cnt = cnt+1



    loss_packets = highest_packet_id+1-cnt
    # print("hightest_packet_id:", highest_packet_id)
    # print("loss_packets_num:",loss_packets)
    # print("loss rate:",loss_packets/(highest_packet_id+1))
    # print("mean_delay:",delaySum/cnt)
    data = {
        "mean_delay": delaySum / cnt,
        "loss rate": loss_packets / (highest_packet_id + 1),
        "delay": delay,
        "rec_id":rec_id,
    }
    return data




def getLinkDelayWithB2B(parameter):
    '''
    从某一个流上得到公共路径上链路的延迟，
    还是按照awk分析方式，即一行行处理，来分析数据，这样复杂度要低一些。
    存储的结构为
    {
    “0-3”：{
            ”0-1“:[0 for i in range(1000)],
            "0-2":[]
            }
    "3-...
    "n+1-n+2"{
            ”0-1“:[],
            "0-2":[]
               }
    }
    :param parameter
    :return:
    '''
    filename = parameter["filename"]
    src = parameter["src"]
    dest1 = parameter["dest1"]
    dest2 = parameter["dest2"]
    sharePathNum = parameter["sharePathNum"]
    ## 构建数据结构
    delay = {}
    for i in range(sharePathNum):
        if i == 0:
            delay["0-3"] = {"0-1":[0 for i in range(1000)],"0-2":[0 for i in range(1000)]}
            continue
        key = str(i+2)+"-"+str(i+3)
        delay[key] = {"0-1":[0 for i in range(1000)],"0-2":[0 for i in range(1000)]}


    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            branchNode = sharePathNum+2
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
            if src == src_ip:
                if dest_ip == dest1:
                    # 放入“0-1”
                    if action == "+":
                        if currentNode == 0:
                            #放入"0-3”当中
                            delay["0-3"]["0-1"][packet_id] = time
                        else:
                            ##放入 其他链路上
                            if currentNode <= sharePathNum+1 and currentNode!=1 and currentNode!=2:
                                key = str(currentNode)+"-"+str(currentNode+1)
                                if key in delay:
                                    delay[key]["0-1"][packet_id] = time
                    if action == "r":
                        if currentNode == 3:
                            #放入"0-3”当中
                            delay["0-3"]["0-1"][packet_id] = time-delay["0-3"]["0-1"][packet_id]
                        else:
                            ##放入 其他链路上
                            if currentNode <= sharePathNum+2 and currentNode!=1 and currentNode!=2:
                                key = str(currentNode-1)+"-"+str(currentNode)
                                if key in delay:
                                    delay[key]["0-1"][packet_id] = time-delay[key]["0-1"][packet_id]
                if dest_ip == dest2:
                    #放入“0-2”
                    if action == "+":
                        if currentNode == 0:
                            #放入"0-3”当中
                            delay["0-3"]["0-2"][packet_id] = time
                        else:
                            ##放入 其他链路上
                            if currentNode <= sharePathNum+1 and currentNode!=1 and currentNode!=2:
                                key = str(currentNode)+"-"+str(currentNode+1)
                                if key in delay:
                                    delay[key]["0-2"][packet_id] = time
                    if action == "r":
                        if currentNode == 3:
                            #放入"0-3”当中
                            delay["0-3"]["0-2"][packet_id] = time-delay["0-3"]["0-2"][packet_id]
                        else:
                            ##放入 其他链路上
                            if currentNode <= sharePathNum+2 and currentNode!=1 and currentNode!=2:
                                key = str(currentNode-1)+"-"+str(currentNode)
                                if key in delay:
                                    delay[key]["0-2"][packet_id] = time-delay[key]["0-2"][packet_id]
    variance = []
    for key in delay:
        temp = copy.copy(delay[key]["0-1"])
        while 0 in temp:
            temp.remove(0)
        temp2 = copy.copy(delay[key]["0-2"])
        while 0 in temp2:
            temp2.remove(0)
        temp.extend(temp2)
        var = Variance_way1(temp)
        variance.append(var)
        print(np.mean(temp))
    print(variance)

    return variance

def Variance_way1(X):
    import numpy as np
    X = np.array(X)
    meanX = np.mean(X)
    n = np.shape(X)[0]
    # 按照协方差公式计算协方差，Note:分母一定是n-1
    variance = sum(np.multiply(X - meanX, X - meanX)) / (n-1)
    return variance

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

def calCovByB2B():
    '''
    从背靠背包的trace文件中计算路径时延，求协方差，和画图
    :return:
    '''
    sharePathNum = [1,2,3,4,5]
    COV = []
    for sPN in sharePathNum:
        cov0 = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/temp/" + str(sPN)
        filename = data_dir + "/cov" + str(sPN)
        if os.path.exists(filename):
            os.remove(filename)
        for cnt in range(100):
            filename = data_dir+'/medium_load'+str(sPN)+"_"+str(cnt)+".tr"
            src = "10.1.1.1"
            dest1 = "10.1."+str(2+sPN)+".2"
            node1 = 1
            data1 = getPathDelay(src, dest1, filename, node1)
            rec_id1 = data1["rec_id"]
            delay1 = data1['delay']
            dest2 = "10.1."+str(2+sPN+2)+".2"
            node2 = 2
            data2 = getPathDelay(src, dest2, filename, node2)
            delay2 = data2['delay']
            rec_id2 = data2["rec_id"]
            hatdelay1 = []
            hatdelay2 = []
            for id in rec_id1:
                if id in rec_id2:
                    index1 = rec_id1.index(id)
                    index2 = rec_id2.index(id)
                    hatdelay1.append(delay1[index1])
                    hatdelay2.append(delay2[index2])
            delay = np.array([hatdelay1, hatdelay2])
            filename = data_dir+"/delay"+str(cnt)
            np.savetxt(filename, delay)
            cov = Covariance_way2(hatdelay1, hatdelay2) ##求协方差
            filename = data_dir+"/cov"+str(sPN)
            open(filename, "a+").write(str(cov) + '\n') ##存入cov文件中
            cov0.append(cov)
        COV.append(np.mean(cov0))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, COV, 'o-', label='Path')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(COV)
    # fig2 = plt.subplot()
    # plt.xlabel('pathNum')
    # plt.ylabel('PC')
    # plt.plot(PN, PC, 'o-', label='HTE')
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    # plt.show()

def calSuccByB2B():
    '''
    从背靠背包的trace文件中计算路径成功传输率，求共享路径成功传输率，和画图
    :return:
    '''
    sharePathNum = [7,8]
    SUCC = []
    for sPN in sharePathNum:
        succ0 = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/medium_load/" + str(sPN)
        filename = data_dir + "/Metric" + str(sPN)
        if os.path.exists(filename):
            os.remove(filename)
        for cnt in range(100):
            filename = data_dir+'/medium_load'+str(sPN)+"_"+str(cnt)+".tr"
            src = "10.1.1.1"
            dest1 = "10.1."+str(2+sPN)+".2"
            node1 = 1
            data1 = getPathDelay(src, dest1, filename, node1)
            rec_id1 = data1["rec_id"]
            dest2 = "10.1."+str(2+sPN+2)+".2"
            node2 = 2
            data2 = getPathDelay(src, dest2, filename, node2)
            rec_id2 = data2["rec_id"]
            succ1 = len(rec_id1)/1000
            succ2 = len(rec_id2)/1000
            succ3 = 0
            for id in rec_id1:
                if id in rec_id2:
                    succ3 = succ3+1
            succ3 = succ3/1000
            succ = [succ1,succ2,succ3]
            filename = data_dir+"/succ"+str(cnt)
            np.savetxt(filename,succ)
            alpha = succ1*succ2/succ3
            filename = data_dir+"/Metric"+str(sPN)
            open(filename, "a+").write(str(alpha) + '\n') ##存入cov文件中
            succ0.append(alpha)
        SUCC.append(np.mean(succ0))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, SUCC, 'o-', label='Path')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(SUCC)
    # fig2 = plt.subplot()
    # plt.xlabel('pathNum')
    # plt.ylabel('PC')
    # plt.plot(PN, PC, 'o-', label='HTE')
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    # plt.show()


def genDelayFileWithB2B():
    '''
    生成指定B2B的Delay文件
    :return:
    '''
    delay_list = list(range(67,100))
    sPN = 6
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/medium_load/" + str(sPN)
    for cnt in delay_list:
        filename = data_dir + '/medium_load' + str(sPN) + "_" + str(cnt) + ".tr"
        src = "10.1.1.1"
        dest1 = "10.1." + str(2 + sPN) + ".2"
        node1 = 1
        data1 = getPathDelay(src, dest1, filename, node1)
        delay1 = data1['delay']
        dest2 = "10.1." + str(2 + sPN + 2) + ".2"
        node2 = 2
        data2 = getPathDelay(src, dest2, filename, node2)
        delay2 = data2['delay']
        assert len(delay1) == len(delay2)
        delay = np.array([delay1, delay2])
        filename = data_dir + "/delay" + str(cnt)
        np.savetxt(filename, delay)
def calCovByB2BFromFile():
    '''

    从calCovByB2B产生的delay文件中读出delay，计算cov,画图
    :return:
    '''
    sharePathNum = [1,2,3,4,5,6,7,8]
    COV = []
    for sPN in sharePathNum:
        cov0 = []
        data_dir = '/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/medium_load/' + str(sPN)
        filename = data_dir + "/cov" + str(sPN)
        if os.path.exists(filename):
            os.remove(filename)
        for cnt in range(100):
            filename = data_dir+"/delay"+str(cnt)
            delay = np.loadtxt(filename)
            # delay = np.dot(delay,100)
            cov = Covariance_way2(delay[0], delay[1])  ##求协方差
            filename = data_dir + "/cov" + str(sPN)
            open(filename, "a+").write(str(cov) + '\n')  ##存入cov文件中
            cov0.append(cov)
        COV.append(np.mean(cov0))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, COV, 'o-', label='Path')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(sharePathNum)

def calMetricByB2BFromFile(load="light_load"):
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    averageMetric = []
    for sPN in sharePathNum:
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/"+load+"/"+str(sPN)
        filename = data_dir + "/cov" + str(sPN)
        cov = np.loadtxt(filename)
        averageMetric.append(np.mean(cov))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, averageMetric, 'o-', label='sandwich')
    filename = data_dir + "/B2B_" + load
    np.savetxt(filename, averageMetric)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(averageMetric)

def getCovByB2BFromFile(load="medium_load"):
    '''

    从calCovByB2B产生的delay文件中读出delay，计算cov,画图
    :return:
    '''
    sharePathNum = [1, 2, 3, 4, 5,6,7,8]
    Label={
        "light_load":"10%|0",
        "medium_load":"39%|0",
        "heavy_load":"97%|4%"
    }
    COV = []
    for sPN in sharePathNum:
        cov0 = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/"+load+"/"+ str(sPN)
        filename = data_dir + "/cov" + str(sPN)
        with open(filename,'r') as f:
            lines = f.readlines()
            for line in lines:
                cov0.append(float(line))
        COV.append(np.mean(cov0))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, COV, 'o-', label=Label[load])
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title("Back-to-back Proing")
    plt.show()
    print(COV)

def calThoughtoutInEveryLink(parameter):
    '''
    {
        "0-3":{
        "starttime":,
        "endtime":,
        "sum_size":
        }
        ...
    }
    粗略计算每一条链路上的带宽
    链路负载比例可以粗略估算为：每条链路上已加载的背景流的带宽和 除以 链路带宽。
    显然这通过增加链路上背景流的数量，可以实现对不同网络负载场景的模拟。
    :return:
    '''
    load = {}
    sharePathNum = parameter["sharePathNum"]
    for i in range(sharePathNum):
        if i == 0:
            load["0-3"] = {}
        else:
            key = str(i+2)+"_"+ str(i+3)
            load[key] = {}
    filename = parameter["filename"]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ", line)
            action = oc[0]
            time = oc[1]
            namespace = oc[2]
            currentNode = int(namespace.split("/")[2])
            length = int(oc[oc.index("length:")+1])
            if action == "r":
                if currentNode == 3:
                    if "starttime" not in load["0-3"]:
                        load["0-3"]["starttime"] = time
                    else:
                        load["0-3"]["endtime"] = time
                    if "sum_size" not in load["0-3"]:
                        load["0-3"]["sum_size"] = length
                    else:
                        load["0-3"]["sum_size"] = load["0-3"]["sum_size"]+length
                else:
                    key = str(currentNode-1) + "_" + str(currentNode)
                    if key in load:
                        if "starttime" not in load[key]:
                            load[key]["starttime"] = time
                        else:
                            load[key]["endtime"] = time
                        if "sum_size" not in load[key]:
                            load[key]["sum_size"] = length
                        else:
                            load[key]["sum_size"] = load[key]["sum_size"] + length
    averageLoad = []
    for key in load:
        load[key]["bandwidth"] = load[key]["sum_size"]/(float(load[key]["endtime"])-float(load[key]["starttime"]))
        print(load[key]["bandwidth"]*8,"bps")
        averageLoad.append(load[key]["bandwidth"]*8)
    print("平均带宽为:",np.mean(averageLoad),"bps")

def calArrivalTimeDiff(parameter):
    '''
    计算三明治包中小包的到达时延之差，相邻的包，奇-偶，如1-0。
    :return:
    '''
    src = parameter["src"]
    dest = parameter["dest"]
    filename = parameter["filename"]
    node = parameter["node"]
    highest_packet_id = 0
    arrivalTime = {}
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
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
            if currentNode == node and src_ip == src and dest_ip == dest and action == "r":
                if packet_id > highest_packet_id:
                    highest_packet_id = packet_id
                    arrivalTime[packet_id] = time
    arrivalInterval = []
    for i in range(highest_packet_id+1):
        if i%2==0 and i+1 in arrivalTime and i in arrivalTime:
            interval = arrivalTime[i+1] - arrivalTime[i]
            arrivalInterval.append(interval)
    # assert highest_packet_id+1 == 2*(len(arrivalInterval)+1)
    data = {
        "arrivalTime":arrivalTime,
        "arrivalInterval":arrivalInterval,
    }
    return data
def calSandWich(dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data/zero_mean/",load="light_load"):
    '''
    从trace文件中计算到达时间差，并画图
    :return:
    '''
    averageMetric = []
    sharePathNum = [1,2,3,4,5,6,7,8]
    for sPN in sharePathNum:
        data_dir = dir + str(sPN)
        filename1 = data_dir + "/arrivalInterval" + str(sPN)
        if os.path.exists(filename1):
            os.remove(filename1)
        filename2 = data_dir + "/arrivalTime" + str(sPN)
        if os.path.exists(filename2):
            os.remove(filename2)
        average = []
        for cnt in range(1):
            filename = data_dir +"/"+ load + str(sPN) + "_" + str(cnt) + ".tr"
            src = "10.1.1.1"
            dest = "10.1." + str(2 + sPN) + ".2"
            node = 1
            parameter = {
                "src":src,
                "dest":dest,
                "filename":filename,
                "node":node
            }
            data = calArrivalTimeDiff(parameter)
            filename3 = data_dir+"/Interval" + str(sPN) + "_" + str(cnt)
            arrivalInterval = data["arrivalInterval"]
            np.savetxt(filename3,arrivalInterval)
            average.append(np.mean(arrivalInterval))
            open(filename1, "a+").write(str(np.mean(arrivalInterval)) + '\n')
        averageMetric.append(np.mean(average))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, averageMetric, 'o-', label='sandwich')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(averageMetric)

def calSandWichFromFile(load="light_load"):
    '''
    从calSandwich生成的文件中画图
    :return:
    '''
    sharePathNum = [1, 2, 3, 4, 5,6,7,8]
    averageMetric = []
    for sPN in sharePathNum:
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich/"+load+"/" + str(sPN)
        filename = data_dir + "/arrivalInterval" + str(sPN)
        arrivalInterval = np.loadtxt(filename)
        averageMetric.append(np.mean(arrivalInterval))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, averageMetric, 'o-', label='sandwich')
    filename = data_dir+"/sandwich_"+load
    np.savetxt(filename,averageMetric)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(averageMetric)

def calLossMetricFromFile(load="heavy_load"):
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    averageMetric = []
    for sPN in sharePathNum:
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/" + load + "/" + str(sPN)
        filename = data_dir + "/Metric" + str(sPN)
        tsr = np.loadtxt(filename)
        averageMetric.append(np.mean(tsr))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, averageMetric, 'o-', label='sandwich')
    filename = data_dir + "/loss" + load
    np.savetxt(filename, averageMetric)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(averageMetric)

def genAverageInterval(load="light_load"):
    '''
    从Interval文件中 计算这一次实验的均值，添加到arrivalInterval中
    :return:
    '''
    sharePathNum = [8]
    for sPN in sharePathNum:
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich/" + load + "/" + str(sPN)
        filename = data_dir + "/arrivalInterval" + str(sPN)
        arrivalInterval = []
        for cnt in range(100):
            filename2 = data_dir + "/Interval" + str(sPN) + "_" + str(cnt)
            Interval = np.loadtxt(filename2)
            arrivalInterval.append(np.mean(Interval))
        np.savetxt(filename,arrivalInterval)
def getTimeInterval():
    '''
    从arrivalInterval 中获取每一次实验的平均到达时间差
    :return:
    '''
    sharePathNum = [1,2,3,4,5,6,7]
    Interval = []
    for sPN in sharePathNum:
        interval = []
        data_dir = '/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich/medium_load/' + str(sPN)
        filename = data_dir + "/arrivalInterval" + str(sPN)
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                interval.append(float(line))
        Interval.append(np.mean(interval))
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, Interval, 'o-', label='Path')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print(Interval)

def calPathlossRate(srcIP="10.1.1.1",destIP="10.1.3.2",sourceNode=0,destNode=1,filename=" "):
    '''
    计算路径丢包率
    :param srcIP:
    :param destIP:
    :param sourceNode:
    :param destNode:
    :return:
    '''
    highest_id = 0
    last_time = 0
    receive_packets = 0
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
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
            if action == "+" and currentNode == sourceNode and src_ip ==srcIP and dest_ip == destIP:
                if packet_id > highest_id:
                    highest_id = packet_id
            elif action == "r" and currentNode == destNode and src_ip ==srcIP and dest_ip == destIP:
                print(packet_id)
                if time > last_time:
                    last_time = time
                receive_packets = receive_packets+1
    print(receive_packets)
    print(highest_id+1)
    print(last_time)
    return receive_packets/(highest_id+1)

def calLinklossRate(srcIP="10.1.1.1",destIP="10.1.12.2",sourceNode=0,destNode=3,filename=" ",action1="r",action2="r"):
    '''
    计算链路的丢包率
    :return:
    '''
    p = [_ for _ in range(1000)]
    highest_id = 0
    last_time = 0
    receive_packets1 = 0
    receive_packets2 = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
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
            if action == action1 and currentNode == sourceNode and src_ip == srcIP and dest_ip == destIP:
                if packet_id > highest_id:
                    highest_id = packet_id
                receive_packets1 = receive_packets1+1
            elif action == action2 and currentNode == destNode and src_ip == srcIP and dest_ip == destIP:
                # print(packet_id)
                if time > last_time:
                    last_time = time
                    receive_packets2 = receive_packets2 + 1
                    p.remove(packet_id)
    # print("receive_packets1:",receive_packets1)
    # print("highest_id:",highest_id)
    # print("receive_packets2:",receive_packets2)
    # print(last_time)
    print(p)
    if sourceNode == 0:
        # print(receive_packets2/(highest_id+1))
        return 1-receive_packets2/(highest_id+1)
    # print((receive_packets2-receive_packets1) / receive_packets1)
    return (receive_packets1-receive_packets2) / receive_packets1

def plot_xy():
    sharePathNum = [1,2,3,4,5,6,7,8]
    Interval = [3.793353620620635e-08, 4.785143739039012e-08, 7.988528839139134e-08, 1.284970245355347e-07, 1.7139601538438422e-07,2.29005191540543e-07, 1.9500083167667698e-07, 2.792227268758791e-07]
    fig1 = plt.subplot()
    plt.xlabel('sharePathNum')
    plt.ylabel('sharePathDistance')
    plt.plot(sharePathNum, Interval, 'o-', label='Path')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

def getSanwichWithLessData(load,num=200):
    '''
    随机选取100组实验中的某个实验中数据的某部分
    :return:
    '''
    sharePathNum = [1,2,3,4, 5, 6, 7,8]
    Interval = []
    for sPN in sharePathNum:
        interval = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich/"+load+"/" + str(sPN)
        filename = data_dir + "/Interval" + str(sPN)+"_"+str(np.random.randint(0,99))
        with open(filename, 'r') as f:
            lines = f.readlines()
            lower = np.random.randint(0,len(lines)-num)
            upper =lower+num
            temp = 0
            for line in lines:
                if temp < lower:
                    temp = temp+1
                    continue
                if temp >= upper:
                    break
                interval.append(float(line))
                temp = temp+1
        Interval.append(np.mean(interval))
    return Interval
    # fig1 = plt.subplot()
    # plt.xlabel('sharePathNum')
    # plt.ylabel('sharePathDistance')
    # plt.plot(sharePathNum, Interval, 'o-', label='Path')
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    # plt.show()
    # print(Interval)

def plot_sandwich():
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich/"
    LOAD = ["light_load","medium_load","heavy_load"]
    LABEL = ["10%|0","39%|0","97%|4%"]
    PATHNUM = [1,2,3,4,5,6,7,8]
    for load in LOAD:
        filename = data_dir+load+"/"+"sandwich"+"_"+load
        K = np.loadtxt(filename)
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LABEL[LOAD.index(load)]
        for i in range(len(K)):
            K[i] = K[i]-0.02
        plt.plot(PATHNUM, K, 'o-', label=label)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Sandwich Probing', loc='center')
    plt.show()

def plot_sandwich_part(num=200):
    LOAD = ["light_load","medium_load","heavy_load"]
    LABEL = ["10%|0","40%|0","97%|4%"]
    PATHNUM = [1,2,3,4,5,6,7,8]
    for load in LOAD:
        K = getSanwichWithLessData(load,num)
        fig1 = plt.subplot()
        plt.xlabel('sharedPathNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        for i in range(len(K)):
            K[i] = K[i]-0.02
        plt.plot(PATHNUM, K, 'o-', label=label)
        # for a, b in zip(PATHNUM, K):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=3)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Sandwich Probing(interarrival time difference)', loc='center')
    plt.show()

def plotB2BCOV():
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/"
    LOAD = ["light_load", "medium_load","heavy_load"]
    LABEL = ["10%|0", "40%|0", "97%|4%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for load in LOAD:
        filename = data_dir + load + "/" + "B2B" + "_" + load
        K = np.loadtxt(filename)
        print(K)
        fig1 = plt.subplot()
        plt.xlabel('sharedPathNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, K, 'o-', label=label)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Back-to-back Proing(delay covariance)', loc='center')
    plt.show()

def plot_loss():
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/"
    LOAD = ["light_load", "medium_load", "heavy_load"]
    LABEL = ["10%|0", "40%|0", "97%|4%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for load in LOAD:
        filename = data_dir + load + "/" + "loss" + "_" + load
        K = np.loadtxt(filename)
        print(K)
        fig1 = plt.subplot()
        plt.xlabel('sharedPathNum')
        plt.ylabel('sharedPathMetric')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, K, 'o-', label=label)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Loss Measurement(successful transport rate)', loc='center')
    plt.show()

def getShareLinkLossRate():
    '''
    输出共享链路上的丢包率，使用背靠背包计算
    :return:
    '''
    LOSS = []
    LOSS.append(calLinklossRate(srcIP="10.1.1.1", destIP="10.1.12.2", sourceNode=0, destNode=3, filename="/media/zongwangz/RealPAN-13438811621/myUbuntu/data/bandwidth_delay_loss/heavy_load/heavy_load8_0.tr", action1="+",
                    action2="r"))
    for i in range(3,10):
        LOSS.append(calLinklossRate(sourceNode=i,destNode=i+1,filename="/media/zongwangz/RealPAN-13438811621/myUbuntu/data/bandwidth_delay_loss/heavy_load/heavy_load8_0.tr",action1="r",action2="r"))
    print(LOSS,np.mean(LOSS))

def plot_loss_part(num=100):
    LOAD = ["light_load", "medium_load", "heavy_load"]
    LABEL = ["10%|0", "39%|0", "97%|4%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for load in LOAD:
        K = getLossWithLessData(load, num)
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, K, 'o-', label=label)
        # for a, b in zip(PATHNUM, K):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=3)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Loss Measurement'+"  "+str(num), loc='center')
    plt.show()
    plt.close()
def getLossWithLessData(load, num):
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    LOSS = []
    for sPN in sharePathNum:
        loss = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/" + load + "/" + str(sPN)
        filename = data_dir + "/Metric" + str(sPN)
        with open(filename, 'r') as f:
            lines = f.readlines()
            lower = np.random.randint(0, len(lines) - num)
            upper = lower + num
            temp = 0
            for line in lines:
                if temp < lower:
                    temp = temp + 1
                    continue
                if temp >= upper:
                    break
                loss.append(float(line))
                temp = temp + 1
        LOSS.append(np.mean(loss))
    return LOSS
def plot_cov_part(num=100):
    LOAD = ["light_load", "medium_load", "heavy_load"]
    LABEL = ["10%|0", "39%|0", "97%|4%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for load in LOAD:
        K = getCovWithLessData(load, num)
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, K, 'o-', label=label)
        plt.show()
        # for a, b in zip(PATHNUM, K):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=3)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Back-to-back Proing'+"  "+str(num), loc='center')
    plt.show()
    plt.close()

def getCovWithLessData(load, num):
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    COV = []
    for sPN in sharePathNum:
        cov = []
        data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b/" + load + "/" + str(sPN)
        filename = data_dir + "/cov" + str(sPN)
        with open(filename, 'r') as f:
            lines = f.readlines()
            lower = np.random.randint(0, len(lines) - num)
            upper = lower + num
            temp = 0
            for line in lines:
                if temp < lower:
                    temp = temp + 1
                    continue
                if temp >= upper:
                    break
                cov.append(float(line))
                temp = temp + 1
        COV.append(np.mean(cov))
    return COV

def calUDPTroughtOut(sharePathNum,filename,sim_start=0,sim_end=30,headNode=0,tailNode=3,isRecord=False):
    '''
    精准的计算一条链路上UDP背景流带宽 IP层
    :param sharePathNum:
    :param filename:
    :param headNode:
    :param tailNode:
    :return:
    '''
    sourceIP = "10.1.1.1"
    destIP = "10.1.1.2"
    if isRecord:
        record_data = {}
    sumData = 0
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ",line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            currentNode = int(namespace.split("/")[2])
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            protocol = oc[39]
            if currentNode == headNode and sourceIP == src_ip and dest_ip == destIP and protocol=="ns3::UdpHeader":
                # print(line)
                size = int(oc[34])
                if action == "-":
                    sumData = sumData+size
                if isRecord:
                    record_data[packet_id] = (time, size)
    print(sumData*8/(sim_end-sim_start),"bps")
    return sumData*8/(sim_end-sim_start)


def calTCPTroughtOut(sharePathNum,filename,sim_start=0,sim_end=30,headNode=0,tailNode=3,isRecord=False):
    '''
        精准的计算一条链路上Tcp背景流带宽 IP层
        :param sharePathNum:
        :param filename:
        :param headNode:
        :param tailNode:
        :return:
        '''
    sourceIP = "10.1.1.1"
    destIP = "10.1.1.2"
    if isRecord:
        record_data = {}
    sumData = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ", line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            currentNode = int(namespace.split("/")[2])
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            protocol = oc[39]
            if currentNode == headNode and sourceIP == src_ip and dest_ip == destIP and protocol == "ns3::TcpHeader":
                # print(line)
                size = int(oc[34])
                if action == "-":
                    sumData = sumData + size
                if isRecord:
                    record_data[packet_id] = (time, size)
            if currentNode == tailNode and sourceIP == dest_ip and destIP == dest_ip and protocol == "ns3::TcpHeader":
                size = int(oc[34])
                if action == "-":
                    sumData = sumData + size
                if isRecord:
                    record_data[packet_id] = (time, size)

    print(sumData * 8 / (sim_end - sim_start), "bps")
    return sumData * 8 / (sim_end - sim_start)

def plotCovWithDot():
    data_dir= "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b"
    sharePathNum = [1,2,3,4,5,6,7,8]
    LOAD = {
        "light_load":["10%|0",1000000],
        "medium_load":["37%|0",10000],
        "heavy_load":["97%|4%",10],
    }
    for load in LOAD:
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LOAD[load]
        for pathNum in sharePathNum:
            filename = data_dir+"/"+load+"/"+str(pathNum)+"/cov"+str(pathNum)
            cov = np.loadtxt(filename)
            for y in cov:
                plt.scatter(pathNum,y*LOAD[load][1],marker=".",c="red")
        filename1 = data_dir + "/" + load + "/B2B_" + load
        metric = np.loadtxt(filename1)
        metric = np.dot(metric,LOAD[load][1])
        plt.plot(sharePathNum, metric, "o-", c="black",label=label)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title("back to back probing  "+load+"  "+str(LOAD[load][1]))
        plt.show()
        plt.close()
def plotLossWithDot():
    data_dir= "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/b2b"
    sharePathNum = [1,2,3,4,5,6,7,8]
    LOAD = {
        # "light_load":"10%|0",
        # "medium_load":"37%|0",
        "heavy_load":"97%|4%",
    }
    for load in LOAD:
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LOAD[load]
        for pathNum in sharePathNum:
            filename = data_dir+"/"+load+"/"+str(pathNum)+"/Metric"+str(pathNum)
            succ = np.loadtxt(filename)
            for y in succ:
                plt.scatter(pathNum,y,marker=".",c="red")
        filename1 = data_dir+"/"+load+"/loss_"+load
        metric = np.loadtxt(filename1)
        plt.plot(sharePathNum,metric,"o-",c="black",label=label)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title("Loss Measurement  "+load)
        plt.show()
        plt.close()

def plotSandwichWithDot():
    data_dir= "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/sandwich"
    sharePathNum = [1,2,3,4,5,6,7,8]
    LOAD = {
        "light_load":"10%|0",
        "medium_load":"37%|0",
        "heavy_load":"97%|4%",
    }
    for load in LOAD:
        metric = [0 for _ in range(len(sharePathNum))]
        fig1 = plt.subplot()
        plt.xlabel('sharePathNum')
        plt.ylabel('sharePathMetric')
        label = LOAD[load]
        for pathNum in sharePathNum:
            for i in range(100):
                filename = data_dir+"/"+load+"/"+str(pathNum)+"/Interval"+str(pathNum)+"_"+str(i)
                Interval  = np.loadtxt(filename)
                interval = Interval[np.random.randint(0,len(Interval))]
                plt.scatter(pathNum,interval,marker=".",c="red")
                metric[pathNum-1] += interval
        metric = np.dot(metric,1/100)
        plt.plot(sharePathNum,metric,"o-",c="black",label=label)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title("sandwich  "+load)
        plt.show()
        plt.close()

def cal1For():
    '''
    计算调查用的三明治包的数据
    :return:
    '''
    LOAD = {
        "light_load": "10%|0",
        "medium_load": "37%|0",
        "heavy_load": "97%|4%",
    }
    Interval = [0, 5, 10, 20, 50, 100]
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/interval_investigate"
    for load in LOAD:
        for interval in Interval:
            for pathNum in sharePathNum:
                filename = data_dir+"/"+load+"/"+str(interval)+"/"+load+str(pathNum)+"_0_"+str(interval)+".tr"
                src = "10.1.1.1"
                dest = "10.1." + str(2 + pathNum) + ".2"
                node = 1
                parameter = {
                    "src": src,
                    "dest": dest,
                    "filename": filename,
                    "node": node
                }
                data = calArrivalTimeDiff(parameter)
                filename3 = data_dir+"/"+load+"/"+str(interval)+"/"+"Interval"+str(pathNum)+"_0_"+str(interval)
                arrivalInterval = data["arrivalInterval"]
                np.savetxt(filename3, arrivalInterval)

def sandwichInterval(num=100,savefile=False):
    LOAD = {
        "light_load": "10%|0",
        "medium_load": "40%|0",
        "heavy_load": "97%|4%",
    }
    Interval = [0,5,10,20,50,100]
    sharePathNum = [1,2,3,4,5,6,7,8]
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/interval_investigate"
    for load in LOAD:
        fig1 = plt.subplot()
        plt.xlabel('sharedPathNum')
        plt.ylabel('sharedPathDistance')
        for interval in Interval:
            metric = []
            for pathNum in sharePathNum:
                filename = data_dir + "/" + load + "/" + str(interval) + "/" + "Interval" + str(pathNum) + "_0_" + str(interval)
                arrivalInterval = np.loadtxt(filename)
                arrivalInterval = np.add(arrivalInterval,-0.001*interval)
                metric.append(np.mean(arrivalInterval[np.random.randint(0,len(arrivalInterval)-100):]))
            plt.plot(sharePathNum,metric,"o-",label="Interval="+str(interval))
            if interval == 0:
                print(metric)
        plt.title("sandwich  "+load+"  "+LOAD[load])
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.show()
        plt.close()

def lastPacketTime(filename="",sourceNode=0,destNode1=1,destNode2=2,sourceIP = "10.1.1.1",destIP1="10.1.10.2",destIP2="10.1.12.2",):
    '''
    得到最后一个探测包到达的时间，两个路径
    :return:
    '''
    lastime1 = 0
    lastpacketId1 = 0
    lastpacketId2 = 0
    lastime2 = 0
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
                if action == "r" and src_ip == sourceIP and dest_ip == destIP1 and currentNode == destNode1:
                    lastime1 = time
                    lastpacketId1 = packet_id
                if action == "r" and src_ip == sourceIP and dest_ip == destIP2 and currentNode == destNode2:
                    lastime2 = time
                    lastpacketId2 = packet_id
            else:
                break
    print(lastime1, lastime2)
    print(lastpacketId1,lastpacketId2)




def dosomething():
    # X = lognorm.rvs(1, 1, size=2000)
    # X = getNormData()
    # X0 = np.log(X)
    # plt.hist(X,200)
    # plt.show()
    x1 = [2.077499798994971586e-02 ,2.149062613065327171e-02 ,2.221507989949748907e-02 ,2.291404874371857914e-02 ,2.363603216080404149e-02 ,2.437648693467335537e-02 ,2.509646231155776216e-02 ,2.580845427135677883e-02]
    x2 = [2.076618190954772000e-02,2.149796331658291801e-02,2.221899296482413369e-02,2.294797437185928934e-02,2.364812814070352265e-02,2.438164773869347379e-02,2.510155628140702325e-02,2.582217336683417450e-02]
    x3 = [0.020780150753768812, 0.02149015075376885, 0.022204924623115557, 0.0229099497487437, 0.0236194974874372, 0.024380502512562796, 0.02509512562814068, 0.02579984924623114]
    fig = plt.subplot()
    sharePathNum = [1,2,3,4,5,6,7,8]
    plt.plot(sharePathNum,x1,label="light_load")
    plt.plot(sharePathNum,x2,label="moderate_load")
    plt.plot(sharePathNum,x3,label="no_traffic_load")
    plt.xlabel("SharePathNum")
    plt.ylabel("SharePathDistanceMetric")
    plt.title("sandwich probing")
    plt.legend()
    plt.show()
# def test_2(filename="/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/test_2.tr"):
#     '''
#     用来测试是否正确的随机发送包
#     :param filename:
#     :return:
#     '''
#     cntNode1 = 0
#     cntNode2 = 0
#     cntNode3 = 0
#     node1 = []
#     node2 = []
#     node3 = []
#     with open(filename,'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             oc = re.split(r"\)|\(| ",line)
#             action = oc[0]
#             time = float(oc[1])
#             namespace = oc[2]
#             currentNode = int(namespace.split("/")[2])
#             packet_id = int(oc[23])
#             src_ip = oc[35]
#             dest_ip = oc[37]
#             src_port = oc[43]
#             dest_port = oc[45]
#             if currentNode == 0 and action == "-":
#                 if dest_ip == "10.4.1.2":
#                     cntNode1 += 1
#                 if dest_ip == "10.4.2.2":
#                     cntNode2 += 1
#                 if dest_ip == "10.4.3.2":
#                     cntNode3 += 1
#             if currentNode == 1 and action == "r":
#                 node1.append(packet_id)
#                 print("Node 1 receive a packet:",packet_id)
#             if currentNode == 2 and action == "r":
#                 node2.append(packet_id)
#                 print("Node 2 receive a packet:",packet_id)
#             if currentNode == 3 and action == "r":
#                 node3.append(packet_id)
#                 print("Node 3 receive a packet:",packet_id)
#     print("node0 send",cntNode1,cntNode2,cntNode3)
#     print("node1 receive:",node1)
#     print("node2 receive:", node1)
#     print("node3 receive:", node1)
def mean_delay(data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data/zero_mean"):
    '''
    看看重载下 无背景流量 有背景流量（数据量少）  有背景流量（数据量多）的情况下，到达时间差曲线
    :param data_dir:
    :return:
    '''
    sharedPathNum = [1,2,3,4,5,6,7,8]
    Interval = []
    for pathNum in sharedPathNum:
        interval = []
        for i in range(100):
            filename = data_dir+"/"+str(pathNum)+"/Interval"+str(pathNum)+"_"+str(i)
            interval.append(np.mean(np.loadtxt(filename)))
        Interval.append(np.mean(interval))
    fig = plt.subplot()
    plt.plot(sharedPathNum,[2.077499798994971586e-02,2.149062613065327171e-02,2.221507989949748907e-02,2.291404874371857914e-02,2.363603216080404149e-02 ,2.437648693467335537e-02 ,2.509646231155776216e-02 ,2.580845427135677883e-02])
    plt.plot(sharedPathNum,[2.081640625631024991e-02,
2.130880738506422956e-02,
2.164699882218681232e-02,
2.216283999526256349e-02,
2.244554191292199435e-02,
2.299727030586371651e-02,
2.357329582797130021e-02,
2.425889243799576184e-02])
    plt.plot(sharedPathNum,Interval)
    plt.show()
def mean_delay_0():
    '''
    在上面一个函数的基础上 看看误差曲线
    :return:
    '''
    sharedPathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    ## no_traffic
    P1 = np.array([0.020780150753768812, 0.02149015075376885, 0.022204924623115557, 0.0229099497487437, 0.0236194974874372, 0.024380502512562796, 0.02509512562814068, 0.02579984924623114])
    ##重载数据量较少
    P2 = np.array([2.081640625631024991e-02,2.130880738506422956e-02,2.164699882218681232e-02,2.216283999526256349e-02,2.244554191292199435e-02,2.299727030586371651e-02,2.357329582797130021e-02,2.425889243799576184e-02])
    #重载数据量较多
    P3 = np.array([0.02074145564094349, 0.02135300585085593, 0.021928874060962077, 0.02250908147128651, 0.023074279343290326, 0.023604078215083993, 0.024211213436535668, 0.024698672972222793])
    fig = plt.subplot()
    # plt.plot(sharedPathNum,P1-0.02,"o-",label="no background traffic")
    # plt.plot(sharedPathNum, P2-0.02,"x-",label = "less data")
    # plt.plot(sharedPathNum,P3-0.02,".-",label = "more data")
    plt.plot(sharedPathNum,P1-P2,".-",label="less data")
    plt.plot(sharedPathNum,P1-P3,"o-",label="more data")
    plt.xlabel("sharedPathNum")
    plt.ylabel("sharedPathDistance")
    plt.title("Sandwich Probing in heavy load(97%|4%)")
    plt.legend()
    plt.show()
    print(P1-P3)

# def test_0():
#     '''
#     过滤得到纯UdpclientServer发送的包
#     :return:
#     '''
#     filename1 = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/heavy_load8_0.tr"
#     filename2 = "/home/zongwangz/PycharmProjects/temp"
#     f = open(filename1,"r")
#     while True:
#         line = f.readline()
#         if line:
#             oc = re.split(r"\)|\(| ", line)
#             if oc[47] == "ns3::SeqTsHeader":
#                 open(filename2,"a+").write(line+"\n")
#         else:
#             break

def print_trace(filename,pid,srcIP,destIP):
    '''
    输出一条流上某个包的trace
    :param filename:
    :param pid:
    :return:
    '''
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            oc = re.split(r"\)|\(| ",line)
            action = oc[0]
            time = float(oc[1])
            namespace = oc[2]
            packet_id = int(oc[23])
            src_ip = oc[35]
            dest_ip = oc[37]
            src_port = oc[43]
            dest_port = oc[45]
            if src_ip == srcIP and dest_ip == destIP and packet_id == pid:
                print(line)



def data_test(data_dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/sandwich/light_load/4"):
    '''
    对数据进行检验
    :return:
    '''
    Interval = []
    for c in range(15):
        filename = data_dir+"/Interarrival"+str(4)+"_"+str(c)
        Interval.extend(np.loadtxt(filename))
    Interval = np.array(Interval)-0.02
    print("gennorm,fit:",stats.gennorm.fit(Interval))
    # print("norm,fit:",stats.norm.fit(Interval))
    print(stats.kstest(Interval,"gennorm", stats.gennorm.fit(Interval)))
    # print(stats.kstest(np.random.normal(size=10000),"norm"))
    print(stats.kstest(stats.gennorm.rvs(0.09314101803056898, 0.0029226548367247366, 4.436697298783252e-17,size=200),"gennorm",stats.gennorm.fit(stats.gennorm.rvs(0.09314101803056898, 0.0029226548367247366, 4.436697298783252e-17,size=200))))

def drawNormHist(X,xlabel="",ylabel="",title=""):
    #创建直方图
    X = np.array(X)
    fig1 = plt.subplot()
    plt.hist(X,bins='auto')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()

    # mu,sigma = norm.fit(X)
    # print("均值: ",mu,"标准差: ",sigma)
    #
    # beta,loc,scale = stats.gennorm.fit(X)
    # print("shape: ",beta, "location: ", loc,"scale: ", scale)
    # print("norm: ",kstest(X,'norm',(mu,sigma)))
    # print("gennorm: ",kstest(X,"gennorm",(beta,loc,scale)))

def getNormData(probe_way="sandwich",load="light_load"):
    x = []
    data_dir = "/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/"+probe_way+"/"+load+"/"+str(8)
    for i in range(100):
        filename = data_dir+"/Interarrival"+str(8)+"_"+str(i)
        if i == 0:

            x = list(np.loadtxt(filename))
        else:
            temp = list(np.loadtxt(filename))
            x.extend(temp)
    x = np.array(x)
    return x


if __name__ == "__main__":
    # test_2("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/test_3.tr")
    # plotB2BCOV()
    # data = getPathDelay("10.1.1.1","10.1.3.2","/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/medium_load1_0.tr",1)
    # print(data)
    # calCovByB2B()
    # calSandWich()
    # plot_sandwich_part()
    # plotSandwichWithDot()
    # plot_loss()
    # print_trace("/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/heavy_load2_0.tr",np.random.randint(0,1000),"10.1.1.1","10.1.4.2")
    x = getNormData(load="light_load")
    drawNormHist(x)