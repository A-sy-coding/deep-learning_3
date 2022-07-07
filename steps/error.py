# step43, 44에서 loss.backward()를 실행했을 때 grad.data가 존재하지 않는다는 오류가 계속 발생하였다.
# dezero.functions.py 파일의 MeansquaredError 함수의 backward()의 return값이 존재하지 않았었다.....

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
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

y_pred = predict(x)
loss = F.mean_squared_error(y, y_pred)

# print(y_pred.data[0]-y[0])
print(0.37560989**2)
print(type(y))
print(type(y_pred))
# loss에서 값을 잘 못구한다.  --> 현재 y_pred와 y값의 type이 다르다.
print(loss)
print(x.shape, W1.shape, b1.shape, y_pred.shape, y.shape, loss.shape)
loss.backward()
print(W1.grad.data)

from dezero.utils import plot_dot_graph
# W1.name = 'W1'
# W2.name = 'W2'
# y_pred.name = 
# b1.name = 'b1'
# b2.name = 'b2'
# plot_dot_graph(loss, verbose=False, to_file='test.png')



# lr = 0.2
# iters = 10000

# for i in range(iters):
#     y_pred = predict(x)
#     loss = F.mean_squared_error(y, y_pred)

#     W1.cleargrad()
#     b1.cleargrad()
#     W2.cleargrad()
#     b2.cleargrad()
#     loss.backward()

#     print(loss)
#     print(W1.data)
#     print(W1.grad)
#     # print(b1.grad)
#     W1.data -= lr * W1.grad.data
#     b1.data -= lr * b1.grad.data
#     W2.data -= lr * W2.grad.data
#     b2.data -= lr * b2.grad.data
#     if i % 1000 == 0:
#         print(loss)


# # Plot
# plt.scatter(x, y, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = predict(t)
# plt.plot(t, y_pred.data, color='r')
# plt.show()