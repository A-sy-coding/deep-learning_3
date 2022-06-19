# step24.py 파일은 최적화 문제에 사용되는 벤치마크 함수를 구현하도록 한다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable

# Sphere 함수 구현
def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

# matyas 함수 구현
def matyas(x , y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z

# Goldstein-Price 함수 구현
def goldstein(x, y):
    z = (1 + (x+y+1)**2 * (19-14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x-3*y)**2 * (18-32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
# z = sphere(x, y)
# z = matyas(x,y)
z = goldstein(x, y)

z.backward() # 역전파 수행
print(x.grad, y.grad)