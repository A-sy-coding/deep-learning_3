# cuda와 관련된 함수들을 정의한다.

import numpy as np

gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from dezero.core import Variable

def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
    
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp

def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)

def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('쿠파이를 로드할 수 없습니다. 쿠파이를 설치해주세요')
    return cp.asarray(x)