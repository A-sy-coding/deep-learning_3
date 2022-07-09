# step46은 매개변수 갱신 클래스를 구현하여 확인하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#########################
# SGD 클래스 이용
#########################
import numpy as np
from dezero import Variable, optimizers
import dezero.functions as F
from dezero.models import MLP

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi* x) + np.random.rand(100,1)

# hyper parameter
lr = 0.2
max_iter = 10000
hidden_size = 10

# model
model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)


# taining
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    
    optimizer.update() # 매개변수 갱신
    if i % 1000 == 0:
        print('-----------{} epochs-------------'.format(i))
        print(loss)
        # for p in model.params():
        #     print(p)
        # print(model._params)
