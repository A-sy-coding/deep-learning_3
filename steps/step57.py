if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero import Variable

x1 = np.random.rand(1,3,7,7)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

# pair 함수 덕분에 튜플로 입력해도 튜플로 반환되고 스칼라로 입력해도 튜플로 반환된다.
x2 = np.random.rand(10,3,7,7)
kernel_size = (5,5)
stride = (1,1)
pad = (0,0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)

# simple conv2
N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3,3)

x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)