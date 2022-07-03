# step40은 numpy의 breadcast 기능을 수행하도록 한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable

x = np.array([1,2,3])
y = np.broadcast_to(x, (2,3))
# print(x)
# print(y)

# utils.py에서 만든 sum_to 실행해보기
# from dezero.utils import sum_to

x = np.array([[1,2,3],[4,5,6]])
# y = sum_to(x, (1,3))
# print(y)


# numpy에서의 broadcast 기능
x0 = np.array([1,2,3])
x1 = np.array([10])
y = x0 + x1  # x1의 원소 10을 x0의 길이만큼 복제하여 각 원소를 더해준다.
# print(y)

# Dezero에서의 broadcast 기능
x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0 + x1  # x1의 원소 10을 x0의 길이만큼 복제하여 각 원소를 더해준다.
# print(y)

# add 역전파 broadcast 확인
x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
# y = x0 + x1
# y = x0 * x1
# y = x0 - x1
y = x0 / x1
print(y)

y.backward()
print(x1.grad)