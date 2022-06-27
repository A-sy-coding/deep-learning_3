# step35는 functions.py 파일에 추가한 tanh 함수의 예제를 보여주고 있다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph = True)

iters = 0

# 고차미분 수행
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph = True)

# 계산그래프 그리기
gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')