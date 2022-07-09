# 매개변수 갱신을 위한 클래스
import numpy as np

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
            self.vs[v_key] = np.zeros_like(param.data)

        # print(self.vs)
        v = self.vs[v_key]
        v *= self.momentum  # alpha*v
        v -= self.lr * param.grad.data
        param.data += v
