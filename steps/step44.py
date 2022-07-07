# step44는 매개변수를 담는 함수를 실행해본다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.


import numpy as np
from dezero import Variable, Parameter

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p

# Variable과 Parameter 클래스를 구분할 수 있다.
# print(isinstance(p, Parameter)) # True
# print(isinstance(x, Parameter)) # False
# print(isinstance(y, Parameter)) # False

# Layer 클래스 확인하기
from dezero.layers import Layer
layer = Layer()

layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'

print(layer._params)  # value의 instance가 Parameter일 때만 set에 저장
print('----------')

# for name in layer._params:
#     print(name, layer.__dict__[name])


#############################
# Layer를 이용한 신경망 구현
############################

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

# 데이터
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

l1 = L.Linear(10) # 출력 크기 지정
l2 = L.Linear(1)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.2
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    
    if i %10 == 0:
        print(loss)
    