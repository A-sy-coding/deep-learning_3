if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
# 계산 그래프 시각화해보기
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

# Goldstein-Price 함수 구현
def goldstein(x, y):
    z = (1 + (x+y+1)**2 * (19-14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x-3*y)**2 * (18-32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x,y)
z.backward()

# 이름 설정
x.name = 'x'
y.name = 'y'
z.name = 'z'

plot_dot_graph(z, verbose=False, to_file = 'goldstein.png')