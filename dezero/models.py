from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L
import numpy as np

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
            setattr(self, '1'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            return self.layers[-1](x)  # 마지막에는 sigmoid 생략


#--- VGG16 신경망
class VGG16(Model):
    # 학습된 가중치
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/'\
                    'releases/download/v0.1/vgg16.npz'
    def __init__(self, pretrained=False):
        super().__init__()

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

        # 신경망 지정
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)
    
    def forward(self, x):
        # 64*3*3 -> activation fucntion : relu
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x,2,2)  # pooling을 통해 shape이 낮아진다. (채널은 유지)
        # 128*3*3 -> activation fucntion : relu
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x,2,2)
        # 256*3*3 -> activation fucntion : relu
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x,2,2)
        # 512*3*3 -> activation fucntion : relu
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x,2,2)
        # 512*3*3 -> activation fucntion : relu
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x,2,2)
        # 4096 - linear
        x = F.reshape(x, (x.shape[0], -1))  # 한줄로 펴기(평탄화)
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)

        return x
    
    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:,:,::-1]  # BGR 순서로 재정렬
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype) # 신경망의 속도와 정확도를 높이기 위한 보정 작업
        image = image.transpose((2,0,1))
        return image