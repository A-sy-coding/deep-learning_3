if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import dezero.datasets
import matplotlib.pyplot as plt
from dezero import DataLoader
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
import dezero

# train_set = dezero.datasets.MNIST(train=True, transform=None)
# test_set = dezero.datasets.MNIST(train=False, transform=None)

# print(len(train_set))
# print(len(test_set))

# x, t = train_set[0]
# plt.imshow(x.reshape(28,28), cmap='gray')
# plt.axis('off')
# plt.show()
# print('label :', t)

# MNIST 학습하기 
max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# model = MLP((hidden_size, 10))
model = MLP((hidden_size,hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

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

    print(f'epoch : {epoch + 1}')
    print(f'train loss : {sum_loss/len(train_set):.3f}\
            accuracy : {sum_acc/len(train_set):.3f}')

    sum_loss, sum_acc = 0,0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    
    print(f'test loss : {sum_loss/len(test_set):.3f}\
            accuracy : {sum_acc/len(test_set):.3f}')