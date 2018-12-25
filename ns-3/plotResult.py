import matplotlib.pyplot as plt


def getResult():

    pathNum = list(range(5, 13))
    T_test_PC = [1.0,1.0,1.0,1.0,1.0,0.99,0.99,0.97]
    T_test_ED = [0,0,0,0,0,0.01,0.02,0.13]
    NJ_PC = [1,1,1,1,1,1,1,1]
    NJ_ED = [0,0,0,0,0,0,0,0]
    HTE0_PC = [1.0,1.0,1.0,1.0,1.0,0.99,0.98,0.95]
    HTE0_ED = [0,0,0,0,0,0.03,0.06,0.22]
    HTE_PC = [1.0,1.0,1.0,1.0,0.97,0.98,0.97,0.93]
    HTE_ED = [0,0,0,0,0.06,0.1,0.13,0.41]
    fig1 = plt.subplot()
    plt.xlabel('pathNum')
    plt.ylabel('pc')

    plt.plot(pathNum, T_test_PC, 'o-', label='T-test')
    plt.plot(pathNum, NJ_PC, '.-', label='RNJ')
    plt.plot(pathNum, HTE0_PC, '--', label='HTE0')
    plt.plot(pathNum, HTE_PC, 'x-',label = 'HTE')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

    fig2 = plt.subplot()
    plt.xlabel('pathNum')
    plt.ylabel('edit distance')

    plt.plot(pathNum, T_test_ED, 'o-', label='T-test')
    plt.plot(pathNum, NJ_ED, '.-', label='RNJ')
    plt.plot(pathNum, HTE0_ED, '--', label='HTE0')
    plt.plot(pathNum, HTE_ED, 'x-',label = 'HTE')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

if __name__ == "__main__":
    getResult()
