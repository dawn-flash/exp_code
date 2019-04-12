'''
@project:exp_code
@author:zongwangz
@time:19-3-11 上午7:59
@email:zongwang.zhang@outlook.com
'''
import copy
def TST(R,M):
    '''

    :param R: 目标节点集，从小到达顺序 [1,2,3,4,5]
    :param M: 三路子拓扑信息
    :return:
    '''
    assert isinstance(R,list)
    R.sort()
    dotR = copy.deepcopy(R)  ##储存剩余目标节点
    hatR = []   ##储存推断拓扑中的目标节点
    hatR.extend(dotR[0:3]) ##选取前三个点形成基础的拓扑 1,2,3
    del dotR[0]
    del dotR[0]
    del dotR[0]
    E = [] ##推断的拓扑
    count = len(R)+1 ##用于编号的
    ##一般M的第一个元组就是(1,2,3,T)
    if M[0][0] == hatR[0] and M[0][1] == hatR[1] and M[0][2] == hatR[2]:
        if M[0][3] == 0:
            E.append((0, count))
            E.append((count, hatR[0]))
            E.append((count, hatR[1]))
            E.append((count, hatR[2]))
            count += 1
        elif M[0][3] == 1:
            E.append((0, count))
            E.append((count, count + 1))
            E.append((count, hatR[2]))
            count += 1
            E.append((count, hatR[0]))
            E.append((count, hatR[1]))
            count = count + 1
        elif M[0][3] == 2:
            E.append((0, count))
            E.append((count, count + 1))
            E.append((count, hatR[1]))
            count += 1
            E.append((count, hatR[0]))
            E.append((count, hatR[2]))
            count = count + 1
        elif M[0][3] == 3:
            E.append((0, count))
            E.append((count, hatR[0]))
            E.append((count, count + 1))
            count += 1
            E.append((count, hatR[2]))
            E.append((count, hatR[1]))
            count = count + 1


if __name__ == "__main__":
    VTree = [5, 7, 7, 6, 0, 5, 6]
    E = [(5, 1), (0, 5), (7, 2), (6, 7), (5, 6), (7, 3), (6, 4)]
    R = [1,2,3,4]
    M = [(1, 2, 3, 3),
         (1, 2, 4, 3),
         (1, 3, 2, 3),
         (1, 3, 4, 3)
        , (1, 4, 2, 3)
        , (1, 4, 3, 3)
        , (2, 1, 3, 2)
        , (2, 1, 4, 2)
        , (2, 3, 1, 1),
         (2, 3, 4, 1),
         (2, 4, 1, 1),
         (2, 4, 3, 2),
         (3, 1, 2, 2),
         (3, 1, 4, 2),
         (3, 2, 1, 1),
         (3, 2, 4, 1),
         (3, 4, 1, 1),
         (3, 4, 2, 2),
         (4, 1, 2, 2),
         (4, 1, 3, 2),
         (4, 2, 1, 1),
         (4, 2, 3, 3),
         (4, 3, 1, 1),
         (4, 3, 2, 3)]
    TST(R,M)
