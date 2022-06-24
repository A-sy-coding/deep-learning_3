# 2차 미분을 수행하는 함수 예제를 확인해보고, 오류 수정

if '__file__' in globals():  # 전역 파일이 존재하는지 확인한다.
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph = True)
print(x.grad) # 첫번째 미분값

# 두번째 미분값
gx = x.grad # 첫번째 미분값
gx.backward()  # gx에 미분값이 남아 있는 상태에서 새로운 역전파를 수행
print(x.grad)  # 68이 나오는데,  기존에 있었던 미분값 24에 2차 미분결과 44가 더해져서 나오게 된다.

# 따라서 위의 문제를 해결하기 위해서는 기존에 있었던 미분값을 재설정 해주어야 한다.
x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph = True)
print(x.grad)  # 1차 미분 값 출력

gx = x.grad
x.cleargrad() # 미분값 재설정
gx.backward()
print(x.grad)

####################
# 뉴턴 방법을 활용한 최적화 --> 이전 step들에서는 고차미분이 불가능 했기 때문에 뉴턴 방법이 사용했던 이차미분을 사용할 수 없었다.
# step29.py 파일에서는 이차미분 함수를 직접 만든 뒤 이차 미분을 수행하였다.
x = Variable(np.array(2.0))
iters = 10  # 반복 횟수

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph = True) # 역전파 수행

    gx = x.grad  # 1차 미분한 값
    x.cleargrad() # 미분값 재설정
    gx.backward()  # 2차 미분
    gx2 = x.grad
    
    # 값 갱신
    x.data -= gx.data / gx2.data
