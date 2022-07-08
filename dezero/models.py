from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file = to_file)  # 계산 그래프 생성
        
# 완전연결계층 신경망 구현
class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid): # 인자로 함수가 들어올 수 있다.
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, '1', str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            return self.layers[-1](x)  # 마지막에는 sigmoid 생략


