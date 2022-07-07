################  오류 해결 필요 grad값이 None으로 계속 나온다. ###############3


# if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

# # step43에서는 비선형 데이터셋을 사용하여 문제를 해결하려고 한다.
# # 비선형 데이터의 경우에는 회귀 직선으로는 문제 해결이 불가능하다. -> 따라서 신경망을 이용하여 해결한다.
# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(0)
# x = np.random.rand(100,1)  # (100,1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

# plt.scatter(x,y)
# # plt.show()


# # 신경망 학습
# import numpy as np
# from dezero import Variable
# import dezero.functions as F

# # 데이터셋 준비
# np.random.seed(0)
# x = np.random.rand(5,1)  # (100,1)
# y = np.sin(2 * np.pi * x) + np.random.rand(5,1)

# # 가중치 초기화
# I, H, O = 1, 3, 1 # input, hidden, output
# W1 = Variable(0.01 * np.random.randn(I,H))  # input으로 (I,H)
# b1 = Variable(np.zeros(H))
# W2 = Variable(0.01 * np.random.randn(H, O))  # output으로 (H,O)
# b2 = Variable(np.zeros(O))

# # 신경망 예측
# def predict(x):
#     y = F.linear(x, W1, b1)
#     y = F.sigmoid(y)
#     y = F.linear(y, W2, b2)
#     # y = F.sigmoid_simple(y)
#     # y = F.linear(y, W2, b2)
#     return y

# lr = 0.2
# iters = 1000


# # 신경망 학습
# for i in range(iters):
#     y_pred = predict(x)
#     loss = F.mean_squared_error(y, y_pred)

#     W1.cleargrad()
#     b1.cleargrad()
#     W2.cleargrad()
#     b2.cleargrad()
#     loss.backward()  # 역전파 수행
    
#     # print(W1.data)
#     # print(W1.grad)
#     W1.data -= lr * W1.grad.data
#     b1.data -= lr * b1.grad.data
#     W2.data -= lr * W2.grad.data
#     b2.data -= lr * b2.grad.data

#     if i % 100 == 0:  # 100회마다 출력
#         print(loss)


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


# W1.name = 'W1'
# W2.name = 'W2'
# y_pred.name = 
# b1.name = 'b1'
# b2.name = 'b2'
# plot_dot_graph(loss, verbose=False, to_file='test.png')



lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    # print(b1.grad)
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)


# Plot
# plt.scatter(x, y, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = predict(t)
# plt.plot(t, y_pred.data, color='r')
# plt.show()