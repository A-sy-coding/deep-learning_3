# step23.py는 dezero 패키지를 잘 import하는지 확인하기 위한 테스트 작업이다.

if '__file__' in globals(): # 전역 파일이 존재하는지 확인한다.
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 부모 디렉토리까지 경로를 추가한다.

import numpy as np
from dezero import Variable  # 부모 디렉토리까지 경로를 추가했으므로 dezero를 바로 불러올 수 있다.

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)