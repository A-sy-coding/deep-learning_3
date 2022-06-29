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

# x.data.shape와 x.grad.shape가 일치하도록 반환하는 reshape 함수 구현
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape  # 역전파에서 사용하기 위해 형태를 미리 기억해 놓는다.
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

# dezero용 reshape 함수
from dezero.core import as_variable

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)  # x가 ndarry 형태여도 Variable 인스턴스로 반환한다.
    return Reshape(shape)(x)

# numpy의 transpose 함수 구현
class Transpose(Function):
    def forward(self,x):
        y = np.transpose(x)  # 전치
        return y

    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)