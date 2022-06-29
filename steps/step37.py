# step37은 tensor를 다루도록 구현한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
import dezero.functions as F
from dezero import Variable

# 밑의 수식은 스칼라에 한해서 함수 값을 구하는 방법이다. 
x = Variable(np.array(1.0))
y = F.sin(x)
# print(y)

# 밑의 수식은 x가 행렬일 경우 함수 값을 구하는 방법이다.
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sin(x)
# print(y)

#################################################
# 지금까지 구현한 Dezero 패키지는 입력 shape와 출력 sahpe가 동일할 때에만 계산이 가능하다.
# tensor에서도 입력 출력값이 행렬이여도 마지막 출력값은 스칼라인 계산 그래프에 대한 역전파를 다룰려고 한다.
# tensor의 원소마다 스칼라로 계산하기 때문에 텐서를 사용한 계산에도 역전파를 올바르게 구현할 수 있음을 확인할 수 있다.

x = Variable(np.array([[1,2,3],[4,5,6]]))
c = Variable(np.array([[10,20,30],[40,50,60]]))
t = x + c
# y = F.sum(t)
print(t)

# 역전파들의 shape과 순전파들의 shape에 대응대는 값들의 shape들은 모두 동일하다.
# ex) x.grad.shape == x.shape

