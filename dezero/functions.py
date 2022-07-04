# 새로운 함수 추가
import numpy as np
from dezero.core import Function
from dezero import utils


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

# sum 함구 구현 -> 행렬을 sum하여 scalar값으로 구한다. + axis와 keepdims 기능도 추가
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis = self.axis, keepdims = self.keepdims)
        return y

    def backward(self, gy):
        # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)  # utils.py파일의 reshpae_sum_backward 함수 사용
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis= None, keepdims=False):
    return Sum(axis, keepdims)(x)



class SumTo(Function):
    def __init__(self, shape):
        self.shape= shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = braodcast_to(gy, self.x_shape)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

# broadcast_to 함수 구현 --> broadcast 구현
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self,gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

# 행렬의 곱과 벡터의 내적을 구하는 함수 구현
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
def matmul(x, W):
    return MatMul()(x,W)
