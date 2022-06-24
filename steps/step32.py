# step32에서느 고차 미분을 구현한다.
# 역전파를 수행할 때 계산 그래프를 만들어주면 고차미분을 수행할 수 있게 된다.
# 즉, Variable 클래스의 grad 변수가 ndarray 인스턴스가 아닌 Variable 인스턴스를 참조하도록 한다. ( Variable 인스턴스에서 계산 그래프가 만들어진다.)

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x **2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph = True) # 역전파 수행 시 계산그래프 생성
print(x.grad)

# 위의 1차미분한 값을 한번 더 미분 -> 2차 미분 수행
gx = x.grad
gx.backward()
print(x.grad)