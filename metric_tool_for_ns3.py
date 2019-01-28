'''
this is the tool for ns3 to test metric
author: zongwang.zhang
mail:zongwang.zhang@outlook.com
'''

'''
for back to back probing
计算和画图分离
'''
from tool import *
from ns3_tool import *
import re
import datetime
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


def calDelayCov(dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/back-to-back/",load="medium_load"):
    '''
    计算路径时延，链路时延，计算路径协方差，计算链路方差，分别保存文件
    :return:
    '''
    sharePathNum = [1,2,3,4,5,6,7,8]
    for sPN in sharePathNum:
        data_dir = dir + load + "/" + str(sPN)
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
                            hatLinkDelay1.append(2*0.15485*(sPN+2))
                        if i in rec_id2:
                            hatLinkDelay2.append(pathDelay2[rec_id2.index(i)])
                        if i not in rec_id2:
                            hatLinkDelay2.append(2*0.15485*(sPN+2))
                pathDelayCov = Covariance_way2(hatLinkDelay1, hatLinkDelay2)
                pathDelay = [hatLinkDelay1,hatLinkDelay2]
                open(filename1,"a+").write(str(pathDelayCov)+'\n')
                filename3 = data_dir + "/pathDelay" + str(sPN) + "_" + str(cnt)
                np.savetxt(filename3,pathDelay)
                LinkDelay = []
                for i in range(sPN):
                    if i == 0:
                        LinkDelay.append(getLinkDelay("10.1.1.1", ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"], filename, 0, 3)["linkDelay"])
                        LinkDelay[-1].extend([2*0.15485 for i in range(2000-len(LinkDelay[-1]))])
                    else:
                        LinkDelay.append(getLinkDelay("10.1.1.1", ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"], filename, i+2, i+3)["linkDelay"])
                        LinkDelay[-1].extend([2 * 0.15485 for i in range(2000 - len(LinkDelay[-1]))])
                linkDelayCov = []
                for linkdelay in LinkDelay:
                    linkDelayCov.append(Covariance_way2(linkdelay,linkdelay))
                filename4 = data_dir + "/linkDelay" + str(sPN) + "_" + str(cnt)
                open(filename4,"w").write(str(LinkDelay)+"\n")
                open(filename2,"a+").write(str(linkDelayCov)+"\n")
                print(pathDelayCov,sum(linkDelayCov))


def calInterarrival(dir="/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/sandwich/",load="heavy_load"):
    sharePathNum = [1,2,4,6,8]
    for sPN in sharePathNum:
        data_dir = dir + load +"/"+ str(sPN)
        for cnt in range(100):
            filename = data_dir + "/" + load + str(sPN) + "_" + str(cnt) + ".tr"
            print(filename)
            src = "10.1.1.1"
            dest = "10.1." + str(2 + sPN) + ".2"
            node = 1
            parameter = {
                "src": src,
                "dest": dest,
                "filename": filename,
                "node": node
            }
            data = calArrivalTimeDiff(parameter)
            filename = data_dir + "/Interarrival" + str(sPN) + "_" + str(cnt)
            arrivalInterval = data["arrivalInterval"]
            if os.path.exists(filename):
                os.remove(filename)
            np.savetxt(filename, arrivalInterval)

def calArrivalTimeDiff(parameter):
    '''
    计算三明治包中小包的到达时延之差，相邻的包，奇-偶，如1-0。
    :return:
    '''
    src = parameter["src"]
    dest = parameter["dest"]
    filename = parameter["filename"]
    node = parameter["node"]
    highest_packet_id = -1
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
                    if len(arrivalTime) == 400:
                        break
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


def getLossInfo(data_dir,load):
    '''
    工具，计算公共路径和非公共路径上的丢包率，计算公共路径上的平均丢包率。
    使用这个函数来选择loss探测的最佳负载
    :return:
    '''
    filename = data_dir+"/log"+"-"+load
    open(filename,"a+").write(str(datetime.datetime.now()))
    open(filename, "a+").write("\n")
    sharePathNum = [1,2,3,4,5,6,7,8]
    for sPN in sharePathNum:
        for c in range(100):
            filename1= data_dir+"/"+str(sPN)+"/"+load+str(sPN)+"_"+str(c)+".tr"
            if os.path.exists(filename1):
                print("/"+load+str(sPN)+"_"+str(c)+".tr","处理结果如下：")
                open(filename, "a+").write("/"+load+str(sPN)+"_"+str(c)+".tr"+" 处理结果如下：")
                open(filename, "a+").write("\n")
                path1rec = 0
                path2rec = 0
                branchrec = 0
                ##首先构造record
                record = {}
                for i in range(sPN):
                    if i == 0:
                        record["0-3"] ={}
                        record["0-3"]["before"] = 2000
                        record["0-3"]["after"] = 0
                    else:
                        record[str(i + 2) + "-" + str(i + 3)] = {}
                        record[str(i+2)+"-"+str(i+3)]["before"] = 0
                        record[str(i + 2) + "-" + str(i + 3)]["after"] = 0
                srcIP = "10.1.1.1"
                destIP = ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"]
                with open(filename1, 'r') as f:
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
                            if src_ip == srcIP and dest_ip in destIP and action == "r" and currentNode <= sPN+2:
                                if currentNode == 1:
                                    path1rec+=1
                                elif currentNode == 2:
                                    path2rec+=1
                                elif currentNode == sPN+2:
                                    branchrec+=1
                                ##为探测包流,
                                for key in record:
                                    str1,str2 = key.split("-")
                                    node1 = int(str1)
                                    node2 = int(str2)
                                    if node1 == currentNode:
                                        record[key]["before"]+=1
                                    if node2 == currentNode:
                                        record[key]["after"]+=1
                        else:
                            break
                average = []
                for key in record:
                    record[key]["loss_rate"] = 1-(record[key]["after"]/record[key]["before"])
                    average.append(record[key]["loss_rate"])
                    print("    ",key,"loss rate:",record[key]["loss_rate"])
                    open(filename, "a+").write(
                        "    "+key+"loss rate:"+str(record[key]["loss_rate"]))
                    open(filename, "a+").write("\n")
                record["path_1"] = path1rec/1000
                record["path_2"] = path2rec/1000
                record["average"] = np.mean(average)
                record["branch"] = branchrec/2000
                open(filename, "a+").write("     path_1:"+str(record["path_1"])+"path_2:"+str(record["path_2"])+"branch:"+str(record["branch"])+"average:"+str(record["average"]))
                open(filename, "a+").write("\n")
                print("     path_1:",record["path_1"],"path_2:",record["path_2"],"branch:",record["branch"],"average:",record["average"])

def getSucc(filename,sPN):
    '''
    calLossRate 的调用函数
    :param filename:
    :param sPN:
    :return:
    '''
    record = {
        "1":[],
        "2":[]
    }
    branchrec = 0
    srcIP = "10.1.1.1"
    destIP = ["10.1." + str(2 + sPN) + ".2", "10.1." + str(4 + sPN) + ".2"]
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
                if src_ip == srcIP and dest_ip in destIP and action == "r" and currentNode <= sPN + 2:
                    if currentNode == 1:
                        record["1"].append(packet_id)
                    elif currentNode == 2:
                        record["2"].append(packet_id)
                    elif currentNode == sPN + 2:
                        branchrec += 1
            else:
                break
    succ12 = 0
    for id in record["1"]:
        if id in record["2"]:
            succ12+=1
    return [len(record["1"])/1000,len(record["2"])/1000,succ12/1000,branchrec/2000]
def calLossRate(data_dir,load):
    '''
    计算理论的成功传输率和实际的传输率,存储为文件
    :param data_dir:
    :param load:
    :return:
    '''
    sharePathNum = [1, 2, 3, 4, 5, 6, 7, 8]
    for sPN in sharePathNum:
        for c in range(100):
            filename1 = data_dir + "/" + str(sPN) + "/" + load + "-"+ str(sPN) + "_" + str(c) + ".tr"
            if os.path.exists(filename1):
                print("/" + load + str(sPN) + "_" + str(c) + ".tr", "处理中：")
                info = getSucc(filename1,sPN)
                print(info)
                filename = data_dir + "/" + str(sPN) + "/result"+str(sPN)+"_"+str(c)
                np.savetxt(filename,info)
    print("完成")

def get_sandwich(data_dir,load):
    '''
    计算三明治一种负载下的曲线
    :param data_dir:
    :param load:
    :return:
    '''
    curve = []
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        Interarrival = []
        for c in range(100):
            filename = data_dir +"/"+load+"/"+str(pathNum)+"/"+"Interarrival"+str(pathNum)+"_"+str(c)
            lines = open(filename,"r").readlines()
            Interarrival.append(float(lines[np.random.randint(0,len(lines))]))
        curve.append(np.mean(Interarrival)-0.02)
        # curve.append(np.log(np.mean(Interarrival) - 0.02))
    return curve
def plot_sandwich(data_dir):
    '''
    plot end to end measurement of sandwich probing
    :param data_dir:
    :return:
    '''
    LOAD = ["no_traffic","light_load","medium_load","heavy_load"]
    LABEL = ["no_traffic","9%|0","40%|0","75%|<1%"]
    PATHNUM = [1,2,3,4,5,6,7,8]
    for load in LOAD:
        curve = get_sandwich(data_dir,load)
        fig1 = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve, 'o-', label=label)
        # for a, b in zip(PATHNUM, K):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=3)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('Sandwich Probing(interarrival time)', loc='center')
    plt.show()

def get_b2b(data_dir, load):
    '''
    得到背靠背包的cov(a,b),和sum_{X(a)}
    :return:
    '''
    curve1 = [] ##cov(a,b)
    curve2 = [] ##sum_{X(a)}
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    # PATHNUM = [1]
    for pathNum in PATHNUM:
        filename1 = data_dir+"/"+load+"/"+str(pathNum)+"/pathDelayCov"+str(pathNum)
        if os.path.exists(filename1):
            curve1.append(np.loadtxt(filename1).mean())
        filename2 = data_dir+"/"+load+"/"+str(pathNum)+"/linkDelayCov"+str(pathNum)
        if os.path.exists(filename2):
            linkDelay = []
            lines = open(filename2,'r').readlines()
            for line in lines:
                linkDelay.append(np.sum(ast.literal_eval(line)))
            curve2.append(np.mean(linkDelay))
    return curve1,curve2

def plot_b2b(data_dir):
    '''
    plot end to end measurement of back to back probing
    :param data_dir:
    :return:
    '''
    # LOAD = ["light_load", "medium_load", "heavy_load"]
    LOAD = ["medium_load"]
    # LABEL = ["9%|0", "40%|0", "75%|1%"]
    LABEL = ["40%|0"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    # PATHNUM = [1]
    for load in LOAD:
        curve1,curve2 = get_b2b(data_dir, load)
        fig = plt.subplots()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label="Covariance")
        plt.plot(PATHNUM,curve2,'x-',label="Sum Variance")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title(label)
        plt.show()
        plt.close()
        print(np.array(curve2)-np.array(curve1))


def get_loss(data_dir,load):
    curve1 = []##真实公共路径传输率
    curve2 = []##计算出来的公共路径传输率
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        loss1 = []
        loss2 = []
        for c in range(100):
            filename = data_dir + "/" + load + "/" + str(pathNum) + "/" + "result" + str(pathNum) + "_" + str(c)
            result = np.loadtxt(filename)
            loss1.append(result[3])
            loss2.append(result[0]*result[1]/result[2])
        curve1.append(np.mean(loss1))
        curve2.append(np.mean(np.array(loss2)))
    return curve1,curve2

def plot_loss(data_dir):
    LOAD = ["heavy_load1", "heavy_load2", "heavy_load3"]
    LABEL = ["75%|1%", "80%%|2%", "85%|4%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    Diff = []
    for load in LOAD:
        curve1,curve2 = get_loss(data_dir, load)
        Diff.append(list(np.array(curve1)-np.array(curve2)))
        fig1 = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label=label+"(real value)",c="blue")
        plt.plot(PATHNUM, curve2, 'o-', label=label+"(calculated value)",c="red")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title('loss probing(successful transmit rate)', loc='center')
        plt.show()
        plt.close()

    fig2 = plt.subplot()
    plt.xlabel('sharedLinkNum')
    plt.ylabel('sharedPathDistance')
    for load in LOAD:
        label = LABEL[LOAD.index(load)]
        index = LOAD.index(load)
        plt.plot(PATHNUM,Diff[index],"x-",label=label)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.title('loss probing(successful transmit rate)', loc='center')
    plt.show()
    plt.close()

def get_sandwichwithdot(data_dir,load):
    '''
    计算三明治一种负载下的曲线,点
    :param data_dir:
    :param load:
    :return:
    '''
    curve = []
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        Interarrival = []
        for c in range(100):
            filename = data_dir +"/"+load+"/"+str(pathNum)+"/"+"Interarrival"+str(pathNum)+"_"+str(c)
            lines = open(filename,"r").readlines()
            Interarrival.append(float(lines[np.random.randint(0,len(lines))])-0.02)
        # curve.append(np.mean(Interarrival)-0.02)
        # curve.append(np.log(np.mean(Interarrival) - 0.02))
        curve.append(Interarrival)
    return curve

def plot_sandwichwithdot(data_dir):
    '''
    plot end to end measurement of sandwich probing
    :param data_dir:
    :return:
    '''
    LOAD = ["no_traffic","light_load","medium_load","heavy_load"]
    # LOAD = ["light_load", "medium_load", "heavy_load"]
    LABEL = ["no_traffic","9%|0","40%|0","75%|<1%"]
    # LABEL = ["9%|0", "40%|0", "75%|<1%"]
    PATHNUM = [1,2,3,4,5,6,7,8]
    for load in LOAD:
        curve1 = get_sandwich(data_dir,load)
        curve2 = get_sandwichwithdot(data_dir,load)
        fig1 = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label=label,c="blue")
        for pathnum in PATHNUM:
            for interval in curve2[PATHNUM.index(pathnum)]:
                plt.scatter(pathnum,interval,marker=".",c="red")
        # for a, b in zip(PATHNUM, K):
        #     plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=3)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title('Sandwich Probing(interarrival time)', loc='center')
        plt.show()
def plot_b2b_2(data_dir):
    '''
        plot end to end measurement of back to back probing
        version2
        :param data_dir:
        :return:
        '''
    LOAD = ["light_load", "medium_load", "heavy_load"]
    LABEL = ["9%|0", "40%|0", "75%|1%"]
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for load in LOAD:
        curve1, curve2 = get_b2b_2(data_dir, load)
        fig = plt.subplots()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label="Covariance")
        plt.plot(PATHNUM, curve2, 'x-', label="Sum Variance")
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title(label)
        plt.show()
        plt.close()

def get_b2b_2(data_dir, load):
    '''
    得到背靠背包的cov(a,b),和sum_{X(a)}
    version2
    :return:
    '''
    curve1 = [] ##cov(a,b)
    curve2 = [] ##sum_{X(a)}
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        filename1 = data_dir+"/"+load+"/"+str(pathNum)+"/pathDelayCov"+str(pathNum)
        cov = []
        for c in range(100):
            filename3 = data_dir+"/"+load+"/"+str(pathNum)+"/pathDelay"+str(pathNum)+"_"+str(c)
            pathDelay = np.loadtxt(filename3)
            pathDelay = list(pathDelay)
            pathDelay[0] = list(pathDelay[0])
            pathDelay[1] = list(pathDelay[1])
            assert isinstance(pathDelay,list)
            pathDelay[0].extend([(pathNum + 2) * 0.02 * 2 for i in range(1000 - len(pathDelay[0]))])
            pathDelay[1].extend([(pathNum + 2) * 0.02 * 2 for i in range(1000 - len(pathDelay[1]))])
            cov.append(Covariance_way2(pathDelay[0],pathDelay[1]))
        curve1.append(np.mean(cov))

        filename2 = data_dir+"/"+load+"/"+str(pathNum)+"/linkDelayCov"+str(pathNum)
        linkDelay = []
        lines = open(filename2,'r').readlines()
        for line in lines:
            linkDelay.append(np.sum(ast.literal_eval(line)))
        curve2.append(np.mean(linkDelay))

    return curve1,curve2

def plot_b2bwithdot(data_dir):
    '''
    b2b 画图 加上点
    :return:
    '''
    LOAD = ["light_load","medium_load","heavy_load"]
    LABEL = ["9%|0","40%|0","75%|1%"]
    PATHNUM = [1,2,3,4,5,6,7,8]
    for load in LOAD:
        curve1,curve2 = get_b2b(data_dir,load)
        curve3,curve4 = get_b2bwithdot(data_dir,load)
        fig1 = plt.subplot()
        plt.xlabel('sharedLinkNum')
        plt.ylabel('sharedPathDistance')
        label = LABEL[LOAD.index(load)]
        plt.plot(PATHNUM, curve1, 'o-', label=label+ "cov(a,b)",c="blue")
        plt.plot(PATHNUM, curve1, 'x-', label=label + "sum_{X(a)}", c="black")
        ##pathdelay
        for pathnum in PATHNUM:
            for delay in curve3[PATHNUM.index(pathnum)]:
                plt.scatter(pathnum,delay,marker=".",c="red")
        ##linkdelay
        # for pathnum in PATHNUM:
        #     for delay in curve4[PATHNUM.index(pathnum)]:
        #         plt.scatter(pathnum, delay, marker="o", c="yellow")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        plt.title('b2b Probing(delay cov)', loc='center')
        plt.show()
def get_b2bwithdot(data_dir,load):
    '''
    与plot_b2bwithdot配套的函数，获取100个点
    :param data_dir:
    :param load:
    :return:
    '''
    curve1 = [] ##pathdelaycov
    curve2 = [] ##linkdelayvar
    PATHNUM = [1, 2, 3, 4, 5, 6, 7, 8]
    for pathNum in PATHNUM:
        filename1 = data_dir + "/" + load + "/" + str(pathNum) + "/" + "pathDelayCov" + str(pathNum)
        Delay1 = np.loadtxt(filename1)
        curve1.append(list(Delay1))
        Delay2 = []
        filename2 = data_dir + "/" + load + "/" + str(pathNum) + "/" + "linkDelayCov" + str(pathNum)
        lines = open(filename2,"r").readlines()
        for line in lines:
            Delay2.append(sum(ast.literal_eval(line)))
        curve2.append(Delay2)
    return curve1,curve2

if __name__ == "__main__":
    # srcIP = "10.1.1.1"
    # destIp = "10.1.3.2"
    # filename = "/media/zongwangz/RealPAN-13438811621/myUbuntu/ns3_workspace/NS3/light_load1_0.tr"
    # destnode = 1
    # data = getPathDelay(srcIP,destIp,filename,destnode)
    # print(data)
    # calInterarrival(load="heavy_load")
    # calDelayCov()
    # pass
    # getLossInfo("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/loss_rate/heavy_load-1","heavy_load-1")
    # getLossInfo("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/loss_rate/heavy_load-2", "heavy_load-2")
    # getLossInfo("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/loss_rate/heavy_load-3", "heavy_load-3")
    # plot_sandwich("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/sandwich")
    # calLossRate("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/loss_rate/heavy_load-2","heavy_load-2")
    plot_b2bwithdot("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/back-to-back")
    # calInterarrival(load="no_traffic")
    # plot_sandwich("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/sandwich")
    plot_loss()
    # plot_sandwichwithdot("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/sandwich")
    # plot_loss("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/end_to_end/loss_rate")