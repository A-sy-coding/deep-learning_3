if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.datasets import SinCurve
from dezero.dataloaders import SeqDataLoader

# batch_size 3만큼 시계열 데이터를 crop해서 진행하도록 하는 예제
# train_set = SinCurve(train=True)
# dataloader = SeqDataLoader(train_set, batch_size = 3)
# x, t = next(dataloader)
# print(x)
# print('-----------')
# print(t)

# LSTM을 이용하여 사인파 데이터 다시 학습
import numpy as np
from dezero.optimizers import Adam
import dezero.functions as F
import dezero.layers as L
from dezero import Model
import dezero
import matplotlib.pyplot as plt

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self,x):
        y = self.rnn(x)
        y = self.fc(y)
        return y

model = BetterRNN(hidden_size, 1)
optimizer = Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count  = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            optimizer.update()
    avg_loss = float(loss.data) / count
    print(f'| epoch {epoch+1} | loss {avg_loss:.3f} ')

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
plt.savefig('LSTM_Sin_graph_predict.png')