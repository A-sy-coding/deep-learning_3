# step34에서는 dezero 패키지의 core.py 모듈에 sin 함수를 추가로 구현하고 해당 함수의 예제를 수행하도록 한다.
if '__file__' in globals():  # 전역 파일이 존재하는지 확인한다.
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable
import dezero.functions as F  # dezero/functions.py 파일에 존재하는 함수 사용 가능

# sin 함수 고차 미분하기
x = Variable(np.array(1.0))
y = F.sin(x)
y.backward(create_graph = True)

for i in range(3):
    gx = x.grad
    x.cleargrad()  # 미분값 재설정
    gx.backward(create_graph =True)
    # print(x.grad)

################################
# 조금 확장하여 그래프도 그리기
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

x = Variable(np.linspace(-7,7,200)) # -7~7까지를 200구간으로 나누기
y = F.sin(x) # sin 함수 적용
y.backward(create_graph=True)

logs = [y.data]  # 미분값들 저장

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# 그래프 그리기
labels = ["y=sin(x)", "y", "y`", "y``", "y```"]
# print(logs)   # array 형태로 1,2,3차 미분 값들이 들어가 있다.
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label = labels[i])
plt.legend(loc='lower right')
plt.show()