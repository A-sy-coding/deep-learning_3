
# step42는 linear regression을 구현하도록 한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np

# 데이터 셋 구현
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * + np.random.rand(100,1)
print(x.shape)
print(y.shape)

# 선형 회귀 구현
from dezero import Variable
import dezero.functions as F

x , y = Variable(x), Variable(y)  # Variable 인스턴스로 변경

W = Variable(np.zeros((1,1)))  # 가중치 생성
b = Variable(np.zeros(1))  # 편향 생성

def predict(x):  # 예측값 구하기
     y = F.matmul(x, W) + b
     return y

def mean_squared_error(x0, x1):  # mse 손실값 함수 생성
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

# 경사 하강법 수행
lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)  # 손살값 구하기

    W.cleargrad()
    b.cleargrad()
    loss.backward()  # 역전파 수행

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)  # 값을 계속 갱신

##########################
# 하지만 위의 mean_squeared_error로 계산 그래프를 만들면 필요없이 메모리를 차지하는 노드가 존재할 수 있다.
# 따라서 mean_squared_error를 class로 정의하여 불필요한 메모리를 줄일 수 있게 된다.

