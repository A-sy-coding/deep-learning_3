import numpy as np
import wakref  # 약한 참조를 만들어주는 라이브러리 -> 참조 카운트를 세지 않는다.

class Variable:
    def __init__(self, data):

        # 데이터 타입이 다르면(ndarray가 아니면) 오류가 발생하게끔 구현
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대를 기록하는 변수 -> 세대를 기록하면 역전파를 수행할때 올바른 순서로 진행할 수 있다.

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대 기록

    def backward(self):
        
        # 초기 grad값을 1.0으로 설정
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 재귀문을 이용한 코드
        # f = self.creator
        # if f is not None:
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     x.backward()

        # 반복문을 이용한 코드
        # funcs = [self.creator] # 처리함수들을 리스트에 저장
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
            # x, y = f.input, f.output     # 옆의 두개의 코드는 함수의 입출력이 하나라고 한정했을 때의 코드이다.
            # x.grad = f.backward(y.grad)
            gys = [output().grad for output in f.outputs]   # 옆의 코드부터는 함수의 입출력이 여러개라고 가정했을 때의 코드이다.
                                                            # outputs에서 ()을 붙혀야지 약한 참조 객체에 접근할 수 있다.
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



    def cleargrad(self):  # grad를 초기화해주는 함수를 추가한다. -> 초기화하지 않으면 값이 중복되어져 연산이 수행되게 된다.
        self.grad = None

class Function:
    # def __call__(self, input):
    #     x = input.data
    #     y = self.forward(x)  # forward함수에서 구체적인 계산이 진행하도록 한다.
    #     output = Variable(as_array(y))  # ndarray로 형변환 해준다.
    #     output.set_creator(self)
    #     self.input = input
    #     self.output = output
    #     return output

    def __call__(self, *inputs):  # 위의 __call__ 함수는 하나의 input값에 대해서 계산하였다.
                                # 이 __call__함수는 여러개의 input값에 대해서 계산하도록 한다.
                                # inputs 앞에 *가 붙으면, 리스트를 사용하는 대신 임의 개수의 인수를 건네 함수를 호출할 수 있다. ex) f(1,2,3,...)
        xs = [x.data for x in inputs]  # inputs는 Variable 클래스로 이뤄져 있는 값이다. (그 중 data값만 가져온다.)
        ys = self.forward(*xs) # xs 앞에 *를 붙히면, 리스트의 원소를 낱개로 풀어서 전달하는 것과 같다. ex) [a,b] -> a, b
        
        if not isinstance(ys, tuple):  # 튜플이 아니면 튜플로 변경
            ys = (ys,)
        
        outputs = [Variable(as_array(y)) for y in ys]  # 리스트인 ys값들을 for문을 이용하여 각각 Variable 클래스로 저장한다.

        self.generation = max([x.generation for x in inputs]) # 가장 최근의 세대 값을 저장한다.
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        # self.outputs = outputs  # Function 클래스와 Variable 클래스 간에 순환 참조가 일어나게 되므로 weakref을 통해 약한 참조로 변경한다.
        self.outputs = [wakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]  # outputs의 길이가 1이면 첫번째 원소를 반환한다.

    # 예외처리를 하여 상속받아 구현해야 한다는 사실을 알려준다.
    def forward(self, x):
        raise NotImplementedError()

    # 역전파 함수 상속
    def backward(self, gy):
        raise NotImplementedError()


# Add 클래스의 forward 함수를 구현한다.
# 이때, Add 클래스는 리스트를 받고 리스트로 반환해야 한다. --> Function 클래스를 수정함으로써 인자로 여러개를 받을 수 있게 됨
class Add(Function):
    def forward(self, x0, x1):  # forward 함수에 인자를 여러개 받아오기 위해서는 Function 클래스의 코드를 수정해야 한다.
        y = x0 + x1
        return y

    def backward(self, gy):  # 덧셈의 역전파는 그대로 흘려보내는 것이다.
                            # 입력이 하나, 출력이 2개가 된다.
        return gy, gy

# add 함수를 만듦으로써 Add 클래스 객체를 만들어주는 것을 생략시킬 수 있다.
def add(x0, x1):
    return Add()(x0, x1)


# Function 클래스를 상속받아 입력값을 제곱하는 클래스 구현
# Function 클래스의 forward 함수를 상속받아 오버라이딩한다.
class Square(Function):
    def forward(self,x):
        y = x**2
        return y
    def backward(self,gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


# exp를 계산하는 클래스 구현
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 클래스들을 가지고 있는 함수 구현
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

# 데이터 타입이 계산 후에도 똑같은 타입을 유지하도록 하는 함수
# 즉, 입력이 scaler이면 ndarray 형태로 다시 변환해준다.
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)

# A = Square()
# B = Exp()
# C = Square()
#
# x = Variable(np.array(0.5))
# a = A(x)
# b = B(a)
# y = C(b)
# print(y.data)
#
# y.grad = np.array(1.0)
# y.backward()
# print(x.grad)
#
# ####### 함수로 만든 후 위의 코드 재작성
# x = Variable(np.array(0.5))
# a = square(x)
# b = exp(a)
# y = square(b)
# print(y.data)
#
# y.grad = np.array(1.0)
# y.backward()
# print(x.grad)

# assert y.creator == C  # y값은 C 함수에 의해 발생되었다.
# assert y.creator.input == b # C 함수의 input으로는 b가 들어갔다.
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a
# assert y.creator.input.creator.input.creator == A
# assert y.creator.input.creator.input.creator.input == x

# 연결시킨 함수로 역전파 도전
# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# print(b.grad)
#
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# print(a.grad)
#
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad = B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(x.grad)

# ndarray 외의 데이터 타입을 받지 못하게 설정 후 실행
# x = Variable(np.array(1.0))
# x = Variable(None)
# x = Variable(1.0)


####################
# 테스트
import unittest

# expected 기울기를 자동으로 구해주는 함수 구현
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data-y0.data)/(2*eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square,x)
        fig = np.allclose(x.grad, num_grad)
        self.assertTrue(fig)
# unittest.main()

