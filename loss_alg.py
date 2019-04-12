# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-3-21 下午8:54
@email:zongwang.zhang@outlook.com
'''
# -*- coding: utf-8 -*-

from tool import *
from sim_tool import *
def doSim(data_dir,getMetric,flag=False):
    '''
    获取度量，执行算法函数，保存推断的E和编辑距离，以及精确度，画图
    :param data_dir:
    :return:
    '''
    if(flag):##推断
        ALG = ["ALT", "RNJ", "T-test","HTE"]
        for alg in ALG:
            filename = data_dir+"/"+alg+"/"+"inferredE_"+alg  ##清理一下
            if os.path.exists(filename):
                os.remove(filename)
        PATHNUM = [i + 3 for i in range(8)]
        VTrees = getVTrees(data_dir+"/Topo_4_3_10")
        sourceE = getSourceEs(data_dir+"/SourceE")
        serial_number = 0
        for VTree in VTrees:
            print(str(serial_number),sourceE[VTrees.index(VTree)])
            # if serial_number == 59:
            #     print("debug")
            # else:
            #     serial_number+=1
            #     continue
            pathNum = PATHNUM[int(serial_number / 100)]
            filename = data_dir + "/" +str(pathNum)+ "/Metric" + str(serial_number)
            if os.path.exists(filename):
                for alg in ALG:
                    if alg == "RNJ":
                        ##处理文件
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R),2000,2000)
                        e = 0.00250627
                        inferredE = RNJ(R,S,e)
                        filename1 = data_dir+"/"+alg+"/inferredE_"+alg
                        open(filename1,"a+").write(str(inferredE)+"\n")
                        print("RNJ:",inferredE)
                    else:
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R))
                        if alg == "ALT":
                            e = 0.00250627
                            inferredBE,dotS = ALT(S,R)
                            inferredE = prune(inferredBE,dotS,R,e)
                            filename2 = data_dir + "/" +alg+ "/inferredE_" + alg
                            open(filename2, "a+").write(str(inferredE) + "\n")
                            print("ALT:", inferredE)
                        elif alg == "HTE":
                            transform(S)
                            inferredE = HTE(S,R)
                            filename3 = data_dir + "/" +alg+ "/inferredE_" + alg
                            open(filename3, "a+").write(str(inferredE) + "\n")
                            print("HTE:", inferredE)
                        elif alg == "T-test":
                            transform(S)
                            inferredBE = ALT(S,R)[0]
                            inferredE = TP(inferredBE,R,S)
                            filename3 = data_dir + "/" + alg+"/inferredE_" + alg
                            open(filename3, "a+").write(str(inferredE) + "\n")
                            print("T-test:", inferredE)
            else:
                ##文件不存在
                print("file not exists:",filename)
                sys.exit(0)

            serial_number = serial_number+1
    FILENAME = evaluation(data_dir,flag)
    plotResult(FILENAME=FILENAME,file=True)


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
                    result = np.log(loss3)-np.log(loss2)-np.log(loss1)
                    SIJ.append(result)
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
                        result = np.log(loss3)-np.log(loss2)-np.log(loss1)
                        temp.append(result)
                    SIJ.append(temp)

        else:
            break
    if K == Norm:
        S = np.zeros((n, n))
        index = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    S[i][j] = S[j][i] = SIJ[index]
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
                        SIJ[index][k] = SIJ[index][k]
                    S[key] = SIJ[index]
                    index = index + 1
    return S


if __name__ == "__main__":
    doSim("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/heavy_load", getMetric3, True)
    # getMetric3("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/heavy_load/3/Metric1",3,2000,200)