'''
分析loss 探测产生的数据
'''
import os
import numpy as np
import re
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

if __name__ == "__main__":
    calLossRate("~/data/")