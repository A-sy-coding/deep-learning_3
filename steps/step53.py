if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# np.save를 통해 ndarray 인스턴스를 외부 파일로 저장할 수 있다.
# np.savez를 통해 여러개의 ndarray 인스턴스를 외부 파일로 저장할 수 있다.
x1 = np.array([1,2,3])
x2 = np.array([4,5,6])
data = {'x1':x1, 'x2':x2}

# np.savez('test.npz', x1=x1, x2=x2)
np.savez('test.npz', **data)

arrays = np.load('test.npz')
x1 = arrays['x1']
x2 = arrays['x2']
print(x1)
print(x2)

# 가변 인자
# *args는 인자값들이 가변적으로 들어갈 수 있다.
# **kwargs는 가변적으로 인자값들이 들어가는데 defualt값들을 가지는 인자들을 가진다.

from dezero.layers import Layer
from dezero.core import Parameter

layer = Layer()
l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))
print(layer._params)
        
# 파라미터 평단화 하기
params_dict = {}
layer._flatten_params(params_dict)
print
(params_dict)

# MNIST 데이터 학습 및 가중치 저장하고 불러오기
import os
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
from dezero.datasets import MNIST

max_epoch = 3
batch_size = 100

train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000,10))
optimizer = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')  # model은 Layer을 상속받았기 때문에 load_weights 함수 사용 가능
    
for epoch in range(max_epoch):
    sum_loss = 0

    for x,t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
    
    print(f'epoch : {epoch+1}, loss : {sum_loss/len(train_set):.4f}')

# 매개변수 저장
model.save_weights('my_mlp_npz')