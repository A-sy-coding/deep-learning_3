from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F


# Layer클래스는 변수를 반환하는 클래스이다. -> 매개변수를 유지한다.
class Layer:
    def __init__(self):
        self._params = set()

    # __setattr__은 인스턴스 변수를 설정할 때 호출되는 특수 메서드이다.
    # name이라는 인스턴스 변수에 값으로 value를 전달한다.
    def __setattr__(self, name ,value):
        # print(name, value)
        if isinstance(value, Parameter):
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
            yield self.__dict__[name]  # value 값
            
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


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

    def _init_W(self):  # 가중치 초기화 함수
        I,O = self.in_size, self.out_size
        W_data = np.random.randn(I,O).astype(self.dtype) * np.sqrt(1 / I)  # 가중치 초기값은 무작위호 설정
        self.W.data = W_data  # self.W는 Parameter 인스턴스이기 때문에 값을 이용하려면 W.data로 출력해야 한다.

    def forward(self, x):
        if self.W.data is None:  # 가중치 in_size 자동 설정
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y