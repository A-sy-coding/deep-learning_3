if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
# import cupy as cp
# import numpy as np

# x = cp.arange(6).reshape(2,3)
# print(x)

# y = x.sum(axis=1)
# print(y)

# n = np.array([1,2,3])
# c = cp.asarray(n)
# assert type(c) == np.ndarray

# gpu로 MNIST 학습하기
import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
import dezero.datasets

# max_epoch = 5
# batch_size = 100

# train_set = dezero.datasets.MNIST(train=True)
# train_loader = DataLoader(train_set, batch_size)
# # print(len(list(train_loader)))

# model = MLP((1000,10))
# optimizer = optimizers.SGD().setup(model)

# # gpu 설정
# if dezero.cuda.gpu_enable:
#     train_loader.to_gpu()
#     model.to_gpu()

# for epoch in range(max_epoch):
#     start = time.time()
#     sum_loss = 0

#     print(epoch+1, '--------------------')
#     for x,t in train_loader:
#         y = model(x)
#         loss = F.softmax_cross_entropy(y,t)        
#         model.cleargrads()
#         loss.backward()
#         optimizer.update()
#         sum_loss += float(loss.data) * len(t)

#     elapsed_time = time.time() - start
#     print('epoch : {}, loss: {:.4f}, time: {:.4f}[sec]'.format(epoch_1, sum_loss/len(train_set), elapsed_time))

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

# gpu 설정
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

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