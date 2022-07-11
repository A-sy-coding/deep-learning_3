# step47 은 소프트맥스 함수 및 교차엔트로피 함수 구현
if '__file__' in  globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from dezero.models import MLP
import numpy as np

# 입력 데이터의 차원 수 2
# 3개의 클래스로 분류
model = MLP((10,3))  # hidden : 10, output : 3

x = np.array([[0.2,-0.4]])
y = model(x)  # 초기값이 random이기 때문에 결과값이 계속 달라진다.
# print(y)

# 소프트맥스 함수 구현
from dezero import Variable, as_variable
import dezero.functions as F

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y

x = np.array([[0.2,-0.4]])
y = model(x)  # 초기값이 random이기 때문에 결과값이 계속 달라진다.
# p = softmax1d(y)
p = F.softmax(y)
# print(y)
# print(p)

# softmax_cross_entropy
x = np.array([[0.2,-0.4],[0.3,0.5],[1.3,-3.2],[2.1,0.3]])
t = np.array([2, 0 ,1, 0])
y = model(x)

loss = F.softmax_cross_entropy(y, t)
print(loss)
# loss = F.softmax_cross_entropy_simple(y,t)
