# 매개변수 갱신을 위한 클래스
import numpy as np
import math
from dezero import cuda

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):  # target 데이터 받기
        self.target = target  # target은 Layer를 상속받았기 때문에 params() 들을 가지고 있다.
        return self
    
    def update(self):
        # None 이외의 매개변수값들을 리스트에 모은다.
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks: # self.hooks에는 전치리들이 들어가 있다.
            f(params)
        
        for param in params:
            self.update_one(param)  # 매개변수 갱신

    def update_one(self, param):
        raise NotImplementedError()  # 상속받아서 update_one을 재정의하려고 한다.

    def add_hook(self, f):
        self.hooks.append(f)  # 전처리 요소들을 hooks에 넣는다.


# SGD 클래스 구현 --> 경사하강법으로 매개변수 갱신
class SGD(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

# Momentum 클래스 구현 --> 모멘텀 기법으로 매개변수 갱신
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        # print(self.vs)
        v = self.vs[v_key]
        v *= self.momentum  # alpha*v
        v -= self.lr * param.grad.data
        param.data += v

#-- Adam 클래스 구현
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)
    
    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1-beta1) * (grad - m)
        v += (1-beta2) * (grad*grad-v)
        param.data -= self.lr * m/(xp.sqrt(v) + eps)