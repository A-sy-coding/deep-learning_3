# step38은 원소별로 계산하지 않는 함수에 대해 실행 해본다.
if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.
    
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y.grad)
print(x.grad)


# Variable 클래스에 reshape 함수 추가 후 실행 -> tuple형 또는 인자값 또는 list형이 들어와도 잘 작동한다.
x = Variable(np.random.rand(1,2,3))
y = x.reshape((2,3))
y1 = x.reshape(2,3)
y2 = x.reshape([2,3])
print(y)
print(y1)
print(y2)

# transpose 해보기
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward()
print(y)  # shape = (3,2)
print(x.grad)  # shape = (2,3)

# Variable 클래스에 transpose 함수 추가 후 실행
x = Variable(np.random.rand(2,3))
y = x.transpose()
y1 = x.T   # Variable 클래스에서 @property 데코레이션을 사용하여 함수 T가 인스턴스로 작동하도록 하였다.
print(y)
print(y1)


#######
A,B,C,D = 1,2,3,4
x = np.random.rand(A,B,C, D)
y = x.transpose()
print('------------')
print(x.shape)
print(y.shape)