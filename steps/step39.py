# step39는 sum을 계산하는 함수를 구현하고 풀어본다.
if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable
import dezero.functions as F

# 벡터인 경우 sum
x = Variable(np.array([1,2,3,4,5,6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad) # 입력 형태와 역전파 후의 출력 형태의 shape은 동일하다.

# 벡터가 아닌 경우 sum
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)  # 1 scalar값을 np.broadcast_to()를 통해 x.shape로 확장시켰다.

# np.sum 기능에는 축을 지정할 수 있다.
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, axis=(0,1))  # tuple과 None도 인자로 받을 수 있다.
print(y)
print(x.shape, '->', y.shape)

# np.sum 기능에는 keepdims라는 기능도 있다. -> 입력과 출력의 차원수를 똑같이 유지해준다.
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, keepdims=True)  # x의 축의 수를 유지해준다.
print(y)
print(y.shape)

# Dezero의 sum 함수 사용해보기
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)  # 세로로 값들을 각각 더해준다.
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
print(x.shape)
y = x.sum(keepdims=True)
print(y.shape)
print(y)  # 하나의 scalar 값으로 나오지만, 차원은 그대로 유지하게 해준다.
