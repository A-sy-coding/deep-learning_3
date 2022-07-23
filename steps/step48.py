if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from dezero import datasets

# x, t = datasets.get_spiral(train=True)
# print(x.shape)
# print(t.shape)

# print(x[10], t[10])
# print(x[110], t[110])

# 다중 클래스 분류 수행
import math
import numpy as np
from dezero import datasets
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# hyper parameter
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# data 읽기
x, t = datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model) # model은 Layer 클래스를 상속받았다. -> 계층들의 params를 가지고 있다.

data_size = len(x)
max_iter = math.ceil(data_size / batch_size) # 최대 반복 횟수

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)  # 인덱스 무작위 섞기
    sum_loss = 0

    for i in range(max_iter):  # 한 에포크마다 미니배치 크기만큼 데이터를 묶어 max_iter만큼 반복
        batch_index = index[i * batch_size : (i+1) * batch_size] # 미니배치
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 매개변수 갱신
        y = model(batch_x) # 예측
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d , loss  %.2f' %(epoch +1, avg_loss))

# import dezero
# with dezero.no_grad():
y_pred = model(x)
print(np.argmax(y_pred.data, axis=1))
print(t)