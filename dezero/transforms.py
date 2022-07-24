# 필요한 전처리 함수들을 정의한다.

import numpy as np
try:
    import Image
except:
    from PIL import Image
# from dezero.utils import pair


# Compose 클래스 정의 -> 여러가지 전처리들을 하나의 파이프라인으로 이어준다.
class Compose:
    '''
    Args:
        list of transforms
    '''
    
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        
        for t in self.transforms:
            img = t(img) # 전처리들을 수행
        return img

    
##########################
# Image transform
##########################

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR': # opencv가 가져오는 이미지는 기본값이 BGR이다.
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b,g,r))
            return img
        else:
            return img.convert(self.mode)

# 사이즈 재설정
class Resize:
    '''
    Args:
        size -> (int 또는 (int,int) ) : output size
        mode (int) -> interpolation
    '''
    
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        self.mode = mode
    
    def __call__(self, img):
        return img.resize(self.size, self.mode)


# 이미지 재설정
class CenterCrop:
    '''
    PIL 이미지를 주어진 size로 재설정

    Args:
        size ( int 또는 (int, int) ) -> output size
    '''
    
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size  # 주어진 사이즈로 이미지를 crop한다.
        left = (W-OH) // 2
        right = W - ( (W-OW) // 2 + (W-OW)%2 )
        up = (H - OH) // 2
        bottom = H - ( (H-OH)//2 + (H-OH)%2 )
        return img.crop((left, up, right, bottom))

# 이미지를 배열로 변경
class ToArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2,0,1) # rgb로 변경
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError

# 배열을 이미지로
class ToPIL:
    def __call__(self, array):
        data = array.transpose(1,2,0)
        return Image.fromarray(data)

class RandomHorizontalFlip:
    pass


##########################
# ndarray transform
##########################

class Normalize:
    '''
    정규화
    Args:
        mean (float or sequence) -> 값또는 채너들의 평균
        std (float or sequence) -> 분산
    '''

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)

        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std

class Flatten:
    def __call__(self, array):
        return array.flatten()

class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)

ToFloat = AsType

class ToInt(AsType):
    def __init__(self, dtype=np.int):
        self.dtype = dtype