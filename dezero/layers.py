from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.utils import pair

# Layer클래스는 변수를 반환하는 클래스이다. -> 매개변수를 유지한다.
class Layer:
    def __init__(self):
        self._params = set()

    # __setattr__은 인스턴스 변수를 설정할 때 호출되는 특수 메서드이다.
    # name이라는 인스턴스 변수에 값으로 value를 전달한다.
    def __setattr__(self, name ,value):
        # print(name, value)
        if isinstance(value, (Parameter, Layer)): # Layer도 추가
            self._params.add(name)
        super().__setattr__(name, value)

    # __call__ 메서드를 만들어 입력받은 인수를 건네 forward 메서드를 호출한다.
    def __call__(self, *inputs):
        outputs = self.forward(*inputs) # forward 수행
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            # yield self.__dict__[name]  # value 값

            obj = self.__dict__[name]
            if isinstance(obj, Layer):  # Layer에서 매개변수 꺼내기
                yield from obj.params() # 재귀
            else:
                yield obj
    
    # params들을 평탄화 시키는 함수
    def _flatten_params(self, params_dict, parent_key=''):
        '''
        params_dict(dictionary 형태) : 평탄화된 파라미터들 저장
        parent_key(str) : 종속된 파라미터 이름을 저장
        '''
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name  # 객체 안에 파라미터가 존재하면 /로 종속됨을 표현하도록 한다.
            
            # obj가 Layer이면 그 안에 존재하는 파라미터도 평탄화하도록 재귀적으로 함수를 호출한다.
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        '''
        모델 가중치 저장
        Args:
            path(경로) -> 파일 저장 경로 지정
        Return:
            file
        '''
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        # print('params_dict : \n', params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        # print('array_dict : \n', array_dict)
        
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e: # interrupt가 발생하면 파일을 삭제하도록 한다.
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        '''
        모델 가중치 불러오기
        Args:
            path(경로) -> 모델 가중치 파일이 존재하는 경로
        '''
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)

        for key, param in params_dict.items():
            param.data = npz[key]

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self): # param을 cpu로 전송
        for param in self.params():
            param.to_cpu()

    def to_gpu(self): # param을 gpu로 전송
        for param in self.params():
            param.to_gpu()

################
# 계층으로서의 Linear 클래스 정의 (함수의 Linear 클래스 아님)
################

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size = None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        # I,O = in_size, out_size
        # W_data = np.random.randn(I,O).astype(dtype) * np.sqrt(1 / I)  # 가중치 초기값은 무작위호 설정
        self.W = Parameter(None, name = 'W')  # Parameter 인스턴스 변수의 이름 W가 self._params에 추가된다.
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name='b')

    def _init_W(self, xp=np):  # 가중치 초기화 함수
        I,O = self.in_size, self.out_size
        W_data = xp.random.randn(I,O).astype(self.dtype) * np.sqrt(1 / I)  # 가중치 초기값은 무작위호 설정
        self.W.data = W_data  # self.W는 Parameter 인스턴스이기 때문에 값을 이용하려면 W.data로 출력해야 한다.

    def forward(self, x):
        if self.W.data is None:  # 가중치 in_size 자동 설정
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
            # self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y

#############
# Conv2d
#############

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, 
                pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)

        return y