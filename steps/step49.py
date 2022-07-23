if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero.datasets
import numpy as np
from dezero.models import MLP
import dezero.functions as F
from dezero import optimizers
import math

# train_set = dezero.datasets.Spiral(train=True)
# print(train_set[0])
# print(len(train_set))

# train_set = dezero.datasets.Spiral()
# batch_index = [0,1,2]
# batch = [train_set[i] for i in batch_index]

# x = np.array([example[0] for example in batch])
# t = np.array([example[1] for example in batch])
# print(x)
# print(t)

# Spiral 클래스를 사용하여 학습 진행
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 10))  # (10,10)  -> 출력값이 10개가 나온다.
optimizer = optimizers.SGD(lr).setup(model)  

data_size = len(train_set)
max_iter = math.ceil(data_size/ batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i+1)*batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f'epoch {epoch+1}, loss {avg_loss:.2f}')