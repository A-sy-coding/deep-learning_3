# core.py는 고차미분이 가능하게 끔 한다.

import numpy as np
import weakref  # 약한 참조를 만들어주는 라이브러리 -> 참조 카운트를 세지 않는다.
import contextlib
import dezero

# 해당 클래스는 역전파를 사용할지 사용하지 않을지를 결정할 수 있다.
# 예를 들어, 신경망에는 학습모드와 추론모드가 있는데, 학습모드의 경우는 역전파가 필요하지만, 추론모드에서는 역전파가 필요없다.
class Config:
    enable_backprop = True
    train = True  

# 역전파를 사용하고 사용하지 않는 것을 with문과 contextlib을 통해 구현
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) # Config 클래스에 있는 name 가져오기
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)  # with문을 빠져나올 때 원래 값으로 복원된다.

def test_mode():
    '''
    train이 아닌 test 모드로 인식하게 해준다.
    Return
        with문 안에서 train=False로 변경
    '''
    return using_config('train', False)

# 최종적으로 역전파를 사용하지 안할지를 결정하는 함수를 구현한다.
def no_grad():
    return using_config('enable_backprop', False)

def test_mode():
    return using_config('train', False)

# 데이터 타입이 계산 후에도 똑같은 타입을 유지하도록 하는 함수
# 즉, 입력이 scaler이면 ndarray 형태로 다시 변환해준다.
def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x

# GPU 대응 추가
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)

class Variable:
    __array_priority__ = 200 # ndarray 인스턴스에도 __add__ 메서드가 존재하는데, Variable 인스턴스의 __add__ 메서드가 먼저 호출하도록 하기 위해서는 우선순위를 설정해야 한다.

    def __init__(self, data, name = None):  # 변수들을 구분하기 위해 이름을 붙여줄 수 있는 name 인자를 추가한다.

        # 데이터 타입이 다르면(ndarray가 아니면) 오류가 발생하게끔 구현
        if data is not None:
            if not isinstance(data, array_types): # array_types을 넣어 cupy.ndarray와 np.ndarray를 사용할 수 있게 한다.
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대를 기록하는 변수 -> 세대를 기록하면 역전파를 수행할때 올바른 순서로 진행할 수 있다.

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대 기록

    def cleargrad(self):  # grad를 초기화해주는 함수를 추가한다. -> 초기화하지 않으면 값이 중복되어져 연산이 수행되게 된다.
        self.grad = None

    def backward(self, retain_grad = False, create_graph = False): # 메모리 사용량을 늘리기 위해 중간부분의 grad들은 삭제하도록 코드를 수정한다.
        
        # 초기 grad값을 1.0으로 설정
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            xp = dezero.cuda.get_array_module(self.data) # xp는 np가 될수도 있고, cp가 될 수도 있다.
            self.grad = Variable(xp.ones_like(self.data)) # 위의 코드는 grad를 ndarray로 만들었지만, 이 코드는 grad를 Variable 인스턴스로 만든다.

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # pop을 통해 함수 한개 가져오기
            gys = [output().grad for output in f.outputs]   # 옆의 코드부터는 함수의 입출력이 여러개라고 가정했을 때의 코드이다.
                                                            # outputs에서 ()을 붙혀야지 약한 참조 객체에 접근할 수 있다.

            with using_config('enable_backprop', create_graph):  # 실제 역전파를 구현하는 구간
                gxs = f.backward(*gys)  # 리스트 값들이 unpack되어 들어가게 한다.
                if not isinstance(gxs, tuple): # 튜플이 아니면 튜플로 변경
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    # x.grad = gx  # 문제 발생 -> 똑같은 변수값이 들어가면 원하지 않는 값이 나옴 ex) 1+1은 2가 나와야 하는데, 1을 그대로 복사하기 때문에 1로 잘못 나오게 된다.
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # 변수값들의 합

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:   # retain_grad가 fasle이면 중간 grad값들을 None으로 바꾼다.
    
                for y in f.outputs:
                    y().grad = None  # y는 약한참조 상태이기 때문에 y()로 참조하도록 한다.

    def to_cpu(self):  # 데이터를 cpu로 전송
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self): # 데이터를 gpu로 전송
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    @property    # 해당 구문 때문에 Varuable 클래스를 마치 인스턴스 변수처럼 사용할 수 있게 된다.
    def shape(self):  # np.array에서 shape를 하면 배열의 형상을 알려주듯이 Variable 클래스에서도 해당 함수를 사용할 수 있도록 해준다.
        return self.data.shape

    # 차원수를 알려주는 함수
    @property
    def ndim(self):
        return self.data.ndim

    # 사이즈를 알려주는 함수
    @property
    def size(self):
        return self.data.size

    # 타입을 할려주는 함수
    @property
    def dtype(self):
        return self.data.dtype

    # len 함수를 사용할 수 있도록 하기
    def __len__(self):
        return len(self.data)

    # print가 출력해주는 문자열을 자신이 원하는대로 정의하려면 __repr__ 메서드를 재정의하면 된다.
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)  # 줄바꿈이 있으면 줄바꿈하고 추가로 공백을 9개를 넣어 가지런하게 만든다.
        return 'variable(' + p + ')'

    # 곱셈 연산자 *를 오버로드 하려고 한다.
    # 예를 들어 a,b가 Variable 객체일 때 mul(a,b)를 a*b로 표현 가능하게끔 하려고 한다.
    def __mul__(self, other):
        return mul(self, other)  # 현재 값과 인자로 들어온 값을 서로 곱하도록 한다.
    def __add__(self, other):
        return add(self, other)

    # rmul과 radd가 정의되어 있지 않기 때문에 2.0*x와 같은 식을 구하려고 하면 float인 2.0에는 mul 메서드가 없기 때문에 오류가 발생하게 된다.
    def __rmul__(self, other):
        return mul(self, other)
    def __radd__(self, other):
        return add(self, other)
    def __neg__(self):
        return neg(self)
    def __sub__(self, other):
        return sub(self, other)
    def __rsub__(self, other):
        return rsub(self, other)
    def __truediv__(self, other):
        return div(self, other)
    def __rtruediv__(self, other):
        return rdiv(self, other)
    def __pow__(self, power):
        return pow(self, power)

    # np.reshape와 같은 계산을 구현하기 위한 Variable reshape 함수를 구현한다.
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): # 한자리만 들어오고, 그 instance가 tuple이거나 list이면 [0]으로 한자리 값을 꺼낸다.
            shape = shape[0]
        return dezero.functions.reshape(self, shape)  # functions.reshape함수에 이미 ndarray를 Variable로 변환하는 함수가 내장되어 있다.

    # np.transpose와 같은 계산을 구현하기 위한 Varibale transpose 함수를 구현한다.
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple,list)) or axes[0] is None:
                axes = axes[0]
        
        return dezero.functions.transpose(self,axes)

    # @property는 인스턴스 변수로사용할 수 있게 해주는 데코레이터이다.  --> transpose 대신 T만을 사용할 수 있다. ex) x.T == x.transpose()
    @property
    def T(self):
        return dezero.functions.transpose(self)

    # sum 함수를 Variable 메서드로 사용할 수 있도록 구현
    def sum(self, axis = None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)


# 들어오는 인자값이 Variable 인스턴스 또는 ndarray 인스턴스일 때 반환값을 Variable 인스턴스로 반환해주는 함수
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    
    def __call__(self, *inputs):  # 위의 __call__ 함수는 하나의 input값에 대해서 계산하였다.
                                # 이 __call__함수는 여러개의 input값에 대해서 계산하도록 한다.
                                # inputs 앞에 *가 붙으면, 리스트를 사용하는 대신 임의 개수의 인수를 건네 함수를 호출할 수 있다. ex) f(1,2,3,...)
        inputs = [as_variable(x) for x in inputs]  # 인자로 들어온 값들이 ndarray 또는 Variable일 때 해당 값들을 Variable 인스턴스로 변환해준다.

        xs = [x.data for x in inputs]  # inputs는 Variable 클래스로 이뤄져 있는 값이다. (그 중 data값만 가져온다.)
        ys = self.forward(*xs) # xs 앞에 *를 붙히면, 리스트의 원소를 낱개로 풀어서 전달하는 것과 같다. ex) [a,b] -> a, b
        
        if not isinstance(ys, tuple):  # 튜플이 아니면 튜플로 변경
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]  # 리스트인 ys값들을 for문을 이용하여 각각 Variable 클래스로 저장한다.
        

        if Config.enable_backprop: # 역전파를 사용하면 밑의 코드를 수행하고 역전파를 사용하지 않으면 순전파값만 구하게 된다.
            self.generation = max([x.generation for x in inputs]) # 가장 최근의 세대 값을 저장한다.
            
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs # 순전파 때의 결과값을 저장하는 로직
            
            # self.outputs = outputs  # Function 클래스와 Variable 클래스 간에 순환 참조가 일어나게 되므로 weakref을 통해 약한 참조로 변경한다.
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]  # outputs의 길이가 1이면 첫번째 원소를 반환한다.

    
    # 예외처리를 하여 상속받아 구현해야 한다는 사실을 알려준다.
    def forward(self, x):
        raise NotImplementedError()

    # 역전파 함수 상속
    def backward(self, gy):
        raise NotImplementedError()

# 오버로드한 연산자들을 한꺼번에 가져오는 함수
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow



# Add 클래스의 forward 함수를 구현한다.
# 이때, Add 클래스는 리스트를 받고 리스트로 반환해야 한다. --> Function 클래스를 수정함으로써 인자로 여러개를 받을 수 있게 됨
import dezero.functions
class Add(Function):
    def forward(self, x0, x1):  # forward 함수에 인자를 여러개 받아오기 위해서는 Function 클래스의 코드를 수정해야 한다.
        self.x0_shape, self.x1_shape = x0.shape, x1.shape # 두 형상이 다를 때 브로드캐스트용 역전파를 계산
        y = x0 + x1
        return y

    def backward(self, gy):  # 덧셈의 역전파는 그대로 흘려보내는 것이다.
                            # 입력이 하나, 출력이 2개가 된다.
        gx0 , gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# add 함수를 만듦으로써 Add 클래스 객체를 만들어주는 것을 생략시킬 수 있다.
def add(x0, x1):
    # x1 = as_array(x1) # array 형태가 아닌 x1값을 array로 변환해준다 -> 나중에 Function 클래스에서 Variable 객체로 다시 변환된다.
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


# Mul 클래스를 구현 -> 곱셈과 곱셈의 미분을 구현하도록 한다.
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs # 위의 코드는 Variable 인스턴스에서 ndarray 인스턴스인 data를 꺼냈지만, 이 코드는 Variable 인스턴스를 그대로 둔다.
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):  # Mul 클래스를 함수로써 사용가능하게 함
    # x1 = as_array(x1)
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0,x1)

# 부호를 변환해주는 함수 구현
class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self, gy):
        return -gy

def neg(x):  # 부호를 변환해주는 class를 이용하여 부호를 변환해주는 함수 구현
    return Neg()(x)

# 뺄셈을 해주는 클래스 정의
class Sub(Function):
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0,x1):  # 뺄셈을 수행하는 함수
    # x1 = as_array(x1)  # x1값들을 array 형태로 만들고 이후 Variable 클래스에서 Variable형태로 변환한다.
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0,x1)

# 뺄셈의 경우 2.0 - x와 같은 식이 존재하면 __rsub__ 메서드가 호출되게 되는데 2.0-x가 아니라 x-2.0으로 계산하게 되므로 rsub 함수를 재정의해야된다.
# 덧셈은 왼쪽 오른쪽을 구분할 필요가 없지만, 뺄셈의 경우는 왼쪽과 오른쪽의 구분이 필요하다.
def rsub(x0, x1):
    # x1 = as_array(x1)
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1,x0)


# 나눗셈을 계산하는 클래스 구현
class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y
    def backward(self, gy):
        # x0,x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):  # 나눗셈을 계산하는 함수 구현 -> Div 클래스 사용
    # x1 = as_array(x1)
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1):  # 나눗셈도 오른쪽과 왼쪽의 구분이 중요하다. --> rdiv 메소드를 재정의한다.
    # x1 = as_array(x1)
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1,x0)

# 거듭제곱을 계산하는 클래스 구현
class Pow(Function):
    def __init__(self,c):
        self.c = c # 지수

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

def pow(x, c):  # 거듭제곱하는 함수 -> 인자로 밑과 지수를 받는다.
    return Pow(c)(x)


#########################
# 매개변수를 담는 구조 만들기 
#########################

class Parameter(Variable):  # Variable 클래스를 상속받는다.
    pass






