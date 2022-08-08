if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 역 dropout 확인해보기
import numpy as np
import dezero.functions as F
from dezero import test_mode

x = np.ones(5)
print(x)

# 학습시
y = F.dropout(x)
print(y)

# 테스트 시
with test_mode():
    y = F.dropout(x)
    print(y)