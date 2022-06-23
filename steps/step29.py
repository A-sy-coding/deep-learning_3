# step29.py파일에서는 함수를 최적화하는 코드를 구현하도록 한다. -> 뉴턴 방법으로 푸는 최적화
# 경사하강법은 일반적으로 수렴이 느리다는 단점이 존재한다. -> 뉴턴 방법은 수렴을 좀 더 빨리 할 수 있다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable

# 임의의 함수를 이용하여 뉴턴 최적화 적용하기
def f(x):  # 임의의 함수
    y = x**4 - 2 * x ** 2
    return y

def gx2(x):  # 2차 미분한 함수  --> 2차 미분은 자동으로 못구하기 때문에 수동으로 함수 설정
    return 12*x**2 - 4

x = Variable(np.array(2.0))
iters = 10 # 반복 횟수

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.cleargrad()
    y.backward()
    
    x.data -= x.grad / gx2(x.data)

