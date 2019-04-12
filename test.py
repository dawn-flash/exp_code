'''
单元测试文件
@project:exp_code
@author:zongwangz
@time:19-3-11 下午7:45
@email:zongwang.zhang@outlook.com
'''
import unittest
from tool import *
import threading
import time
# class TestAllFunc(unittest.TestCase):
#     def test_VTree1ToVTree2(self):
#         self.assertEquals([6,6,7,7,0,5,5],VTree1ToVTree2([0,1,1,2,2,3,3]))
'''
suit = unittest.TestSuite()
suit.addTest(TestAllFunc("test_test_VTree1ToVTree2"))
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suit)
'''


def Thread_test():
    def fun1():
        for i in range(100):
            time.sleep(1)
            print("1")
    def fun2():
        for i in range(100):
            time.sleep(1)
            print("2")

    threading.Thread(target=fun1).start()
    threading.Thread(target=fun2).start()
if __name__ == "__main__":
    Thread_test()