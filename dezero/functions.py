# 새로운 함수 추가
import numpy as np
from dezero.core import Function

# sin 함수 구현
class Sin(Function):
    def forward(self,x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx 
 
def sin(x):
    return Sin()(x)  # sin 함수 수행

# cos 함수 구현
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -1 * sin(x)
        return gx

def cos(x):
    return Cos()(x) # cos 함수 수행


# tanh 함수 구현
# tanh 함수를 미분하면 1-tanh^2 이 된다.
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1-y**2)
        return gx

def tanh(x):
    return Tanh()(x)