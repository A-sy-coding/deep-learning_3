# step41은 행렬의 곱과 내적을 구현하도록 한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
# numpy를 이용하여 행렬의 곱과 벡터의 내적 구해보기
import numpy as np

# 벡터의 내적
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.dot(a,b)
# print(c)

# 행렬의 곱
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = np.dot(a,b)
# print(c)

# Variable 인스턴스로 행렬의 곱 실행
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))
y = F.matmul(x, W)
y.backward()

print(y)
print(x.grad.shape)  # shape을 똑같이 유지하고 있다.
print(W.grad.shape)