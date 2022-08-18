if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.layers as L
from dezero.models import Simple_RNN
import dezero.functions as F
from dezero.datasets import SinCurve
import matplotlib.pyplot as plt
import dezero.optimizers

# rnn = L.RNN(10)
# x = np.random.rand(1,1)
# h = rnn(x)
# print(h.shape)

# 간단한 RNN 시도
seq_data = [np.random.randn(1,1) for _ in range(1000)]
# print(seq_data)
xs = seq_data[0:-1]
ts = seq_data[1:]  # 정답 데이터 -> 한 시점 앞선 데이터
# print(xs)

model = Simple_RNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)

    cnt+=1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break
print(loss)

# 사인파 예측
# train_set = SinCurve(train=True)  # (data, label)이 999개가 존재한다.
# print(len(list(train_set)))
# print(len(train_set))
# print(train_set[0])
# print(train_set[1])
# print(train_set[2])

# 그래프 그리기
# xs = [example[0] for example in train_set]
# ts = [example[1] for example in train_set]
# plt.plot(np.arange(len(xs)), xs, label='xs')
# plt.plot(np.arange(len(ts)), ts, label='ts')
# plt.savefig('SinCurve.png')

# RNN 아용하여 사인파 학습
max_epoch = 100
hidden_size = 100
bptt_length = 30
train_set = SinCurve(train=True)  # (data, label)이 999개가 존재한다.
seqlen = len(train_set)

model = Simple_RNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

# 학습시작
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0,0

    for x, t in train_set:
        x = x.reshape(1,1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length ==0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print(f'| epoch {epoch+1} | loss {avg_loss:3f}')

# 예측값 출력
xs = np.cos(np.linspace(0, 4*np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1,1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Sin_graph_predict.png')