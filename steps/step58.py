if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.models import VGG16
import dezero
from PIL import Image
import dezero.datasets

# model = VGG16(pretrained=True)

# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
# model.plot(x)  # 계산 그래프 그리기

# test 해보기
url = 'https://github.com/WegraLee/deep-learning-from-scratch-3/'\
        'raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
x = VGG16.preprocess(img) # 전처리 -> VGG16 input 사이즈로 변경  (3, 224, 224)
x = x[np.newaxis]  # 1차원 추가 (1, 3, 224, 224)

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)
print(predict_id)

model.plot(x, to_file='vgg.png')  # 계산그래프 시각화
labels = dezero.datasets.ImageNet.labels()
# print(labels)
# print(labels[predict_id])
print([k for k,v in labels.items() if v=='zebra'])




# print(type(x), x.shape)
# img.save('ex.jpg')