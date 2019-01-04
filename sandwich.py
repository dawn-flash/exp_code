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