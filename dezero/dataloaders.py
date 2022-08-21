# 데이터를 배치 사이즈만큼 가져오는 DataLoader 클래스 구현

import math
import random
import numpy as np
from dezero import cuda

class DataLoader:
    '''
    배치 사이즈만큼 데이터 가져오기
    Args:
        dataset (ndarray)
        batch_size (int) -> 배치 크기
        shuffle (boolean) -> 섞는 옵션
    '''
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu # gpu설정(default는 false로 되어 있다.)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset)) # index값들 섞기
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size : (i+1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np # self.gpu가 true이면 cupy를 부르고, 아니면 numpy를 부른다.
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True

#-- 시계열용 데이터 로더
class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self):
        '''
        DataLoader 클래스의 __next__함수 오버라이딩
        ''' 
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i*jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = np.array([example[0] for example in batch])  # 훈련 데이터
        t = np.array([example[1] for example in batch])  # 정답 데이터

        self.iteration += 1
        return x, t