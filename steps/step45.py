# step45는 Layer 안에 Layer가 들어갈 수 있도록 확장시킨다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# print(os.path.join(os.path.dirname(__file__)), '..')

# Layer안에 있는 Layer 안에 있는 매개변수 가져오기
# import dezero.layers as L
# import dezero.functions as F
# from dezero import Layer

# model = Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(3)

# print(model._params)  # 매개변수 이름 저장
# print(model.__dict__['l2'])

# # 예측 함수
# def predict(model, x):
#     y = model.l1(x)
#     y = F.sigmoid(y)
#     y = model.l2(y)
#     return y

# # 모든 매개변수에 접근
# print(model._params)
# for p in model.params():
#     print(p)

# model.cleargrads()


#######################
# TowLayerNet 신경망 만들기 -> models.py 파일을 사용하여 Layer 계층을 상속받으면서 계산그래프도 그리도록 한다.
#######################
# import numpy as np
# from dezero import Variable, Model
# import dezero.layers as L
# import dezero.functions as F

# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y

# x = Variable(np.random.randn(5,10), name='x')
# model = TwoLayerNet(100, 10)
# model.plot(x)  # Model 클래스에 존재하는 plot 함수 사용


#######################
# Model 클래스를 이용하여 회귀 문제 해결
######################
import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F

# 데이터 생성
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

# hyper parameter 
lr = 0.2
max_iter = 10000
hidden_size = 10

# 모델 정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model =  TwoLayerNet(hidden_size, 1)

# 학습
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print('----------{} epochs------------'.format(i+1000))
        print(loss)