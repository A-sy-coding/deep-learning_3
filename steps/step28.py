# step28.py파일에서는 함수를 최적화하는 코드를 구현하도록 한다. -> 경사하강법을 사용한 최적화

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

# 로젠브록 함수 구현하기
def rosenbrock(x0, x1):
    y = 100 * (x1-x0**2)**2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 학습률
iters = 1000 # 반복 횟수

# 경사하강법 수행하기
for i in range(iters):
    print(x0,x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()  # Variable 인스턴스를 계속 사용하여 미분값을 구하면 grad가 계속 누적되기 때문에 grad를 초기화 시켜주기 위해서 cleargrad()를 사용한다.
    x1.cleargrad()
    y.backward()

    plt.scatter(x0.data,x1.data)
    plt.plot(x0.data,x1.data)

    x0.data -= lr * x0.grad # 경사하강법 공식
    x1.data -= lr * x1.grad

plt.show()
# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)