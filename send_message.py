from twilio.rest import Client
import os
import time
def sendMessage(message):
    account_sid = "ACe65fe670f8f7f503deff8585818f6428"
    auth_token = "e73c46352a171aef5ce6e99001b4d496"
    client = Client(account_sid, auth_token)
    message = client.messages.create( to="+8615623835526", from_="(337)508-0382", body= message)

def detect1(data_dir,num=800):
    c = 0
    for file in os.listdir(data_dir):
        print(file)
    for i in range(num):
        filename1 = data_dir+"sourceTrace"+str(i)+"tr"
        filename2 = data_dir + "Metric" + str(i)
        if os.path.exists(filename1) or os._exists(filename2):
            c+=1
    print("In "+data_dir,str(c),"/",str(num))
    return str(c)+"/"+str(num)
def detect2(data_dir,num=800):
    cnt = 0
    for sLN in [1,2,3,4,5,6,7,8]:
        for c in range(100):
            filename = data_dir+"/"+"heavy_load-2-"+str(sLN)+"_"+str(c)+"tr"
            if os.path.exists(filename):
                cnt+=1
    print("In " + data_dir, str(cnt), "/", str(num))
    return str(cnt) + "/" + str(num)
def supervise(FList):
    while True:
        for filename in FList:
            str = detect1(filename)
            sendMessage(filename+"     "+str)
            time.sleep(24*60*60)
if __name__ == "__main__":
    FList = []
    supervise(FList)
