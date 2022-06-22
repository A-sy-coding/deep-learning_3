# step27에서는 sin함수를 미분하는 코드를 구현하려고 한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Function # dezero 폴더의 __init__.py 파일에서 함수들을 미리 정의하였기 때문에 dezero라는 패키지만 입력해도 함수들이 같이 정의될 수 있다.
from dezero import Variable  # 예시들을 test해보기 위해 import -> 값들을 Variable 타입으로 변환하기 위해
from dezero.utils import plot_dot_graph   # sin 함수 시각화 그래프 그려보기

class Sin(Function): # sin의 미분은 cos이다.
    def forward(self,x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):  # sin 함수를 계산하는 함수
    return Sin()(x)


# 테일러 급수를 사용하여 sin 함수 구현
import math

def my_sin(x, threshold = 0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2*i + 1)  # a=0인 매클로린 전개를 일반화한 공식
        t = c * x ** (2*i+1) 
        y = y + t  # 위의 값들의 합을 구한다.
        if abs(t.data) < threshold:   # 임계값 설정 --> 즉, 반복횟수 설정
            break
    return y


        

x = Variable(np.array(np.pi/4))
# y = sin(x)
y = my_sin(x)
y.backward()

# 이름 설정
x.name = 'x'
y.name = 'y'

plot_dot_graph(y, verbose=False, to_file = 'sin_function.png')  # 시각화 그래프 그리기

print(y.data)
print(x.grad)





