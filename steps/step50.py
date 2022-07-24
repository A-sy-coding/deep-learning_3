if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero.datasets import Spiral
from dezero import DataLoader

# dataloader 사용하기
# batch_size = 10
# max_epoch = 1

# train_set = Spiral(train=True)
# test_set = Spiral(train=False)
# train_loader = DataLoader(train_set, batch_size)  # x, t값들을 반환
# test_loader = DataLoader(test_set, batch_size, shuffle=False)

# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         print(x.shape, t.shape)
#         break

#     for x, t in test_loader:
#         print(x.shape, t.shape)
#         break


import numpy as np
import dezero.functions as F

y = np.array([[0.2,0.8, 0], [0.1,0.9,0], [0.8,0.1,0.1]])
t = np.array([1,2,0])
acc = F.accuracy(y, t)
print(acc)

import dezero
from dezero.models import MLP
from dezero import optimizers
# Spiral 데이터 학습
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

train_loss, train_acc = [], []
test_loss, test_acc = [], []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    print(f'epoch : {epoch+1} ')
    print(f'train loss : {sum_loss/len(train_set):.4f} \
            accuracy : {sum_acc/len(train_set):.4f}')
    train_loss.append(sum_loss/len(train_set))
    train_acc.append(sum_acc/len(train_set))
    
    sum_loss, sum_acc = 0,0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f'test loss : {sum_loss/len(test_set):.4f} \
            accuracy : {sum_acc/len(test_set):.4f}')
    test_loss.append(sum_loss/len(test_set))
    test_acc.append(sum_acc/len(test_set))

import matplotlib.pyplot as plt

x_range = np.arange(max_epoch)
plt.plot(x_range, train_loss, label='train')
plt.plot(x_range, test_loss, label='test')
plt.legend()
plt.savefig('./Spiral_loss_graph.png')

plt.cla()

plt.plot(x_range, train_acc, label='train')
plt.plot(x_range, test_acc, label='test')
plt.legend()
plt.savefig('./Spiral_acc_graph.png')

