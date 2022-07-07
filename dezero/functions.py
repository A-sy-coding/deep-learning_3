# 새로운 함수 추가
import numpy as np
import dezero
from dezero.core import Function, as_variable, Variable, as_array
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

# exp 함수 구현
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)


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
# from dezero.core import as_variable

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

        # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)  # utils.py파일의 reshape_sum_backward 함수 사용
    def backward(self, gy):
        # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
        #                                 self.keepdims)
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

# 계산 그래프의 메모리 감소를 위한 mean_squared_error 클래스 정의
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1

        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0

        return gx0, gx1
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

#################
# 메모리 효율이 좋지 않은 linear함수와 sigmoid 함수
################

# 메모리의 효율성을 높이기 위한 간단한 선형 모형 구하기
# 효율성을 높이기 위해 필요없는 인스턴스를 사용 후에 삭제하도록 한다.
def linear_simple(x, W, b = None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # t를 다 사용했으면 삭제한다.

    return y

# 시그모이드 활성화 함수를 구현한다. --> 하지만 밑의 시그모이드 함수는 메모리 효율이 좋지는 않다.
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

##################
# 메모리 효율을 향상시킨 linear함수와 sigmoid 함수
##################


# linear 클래스 -> 클래스로 정의하면 메모리의 효율성을 높일 수 있다.
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W) 
        if b is not None:
            y += b

        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


# Sigmoid 클래스 구현
class Sigmoid(Function):
    
    def forward(self, x):
        if isinstance(x, Variable):
            x = x.data
            y = 1 / (1 + np.exp(-x))
            return y
        else:
            y = 1 / (1 + np.exp(-x))
            return y
        # y = 1 / (1 + np.exp(-x))
        # return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx
    
def sigmoid(x):
    return Sigmoid()(x)