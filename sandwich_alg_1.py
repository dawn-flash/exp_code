# -*- coding: utf-8 -*-
'''
@project:exp_code
@author:zongwangz
@time:19-3-20 下午6:56
@email:zongwang.zhang@outlook.com
'''
'''
for sandwich_vary_all
'''
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
            pathNum = PATHNUM[int(serial_number / 100)]
            filename = data_dir + "/" +str(pathNum)+ "/Metric" + str(serial_number)
            if os.path.exists(filename):
                for alg in ALG:
                    if alg == "RNJ":
                        ##处理文件
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R),200,200)
                        e = 0.00038134
                        inferredE = RNJ(R,S,e)
                        filename1 = data_dir+"/"+alg+"/inferredE_"+alg
                        open(filename1,"a+").write(str(inferredE)+"\n")
                        print("RNJ:",inferredE)
                    else:
                        R = getLeafNodes(VTree)
                        S = getMetric(filename,len(R))
                        if alg == "ALT":
                            e = 0.00038134
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

if __name__ == "__main__":
    doSim("/media/zongwangz/RealPAN-13438811621/myUbuntu/data2/alg/sandwich_vary_traffic", getMetric, True)
