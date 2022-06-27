# step36은 double backprop에 대해 설명하고 있다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
import numpy as np
from dezero import Variable

# y = x**2
# z = (dy/dx)**3 + y
# dz 구하기
x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph = True)
gx = x.grad
x.cleargrad()  # grad 초기화

# 이때, gx=x.grad는 단순한 값이 아니라 계산 그래프 식이다.
# 따라서, x.grad의 계산 그래프에 대해 추가로 역전파가 가능하다.

# 두번 역전파 수행
z = gx ** 3 + y
z.backward()
print(x.grad)