import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to

#####################
# conv2_simple / Conv2d 구현
#####################
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    '''
    간단한 conv2 계산
    Args:
        x (Variable or ndarray) : 입력값
        W (Variable or ndarray) : 가중치
        b () : 편향
        stride(int or tuple) : stride 폭
        pad(int or tuple) : 패딩 정도
    Return:
        예측값
    '''
    x, W = as_variable(x), as_variable(W) # Variable로 변환

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape  # 커널
    SH, SW = pair(stride)
    PW, PH = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose() # 2차원으로 변경
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0,3,1,2)
    return y

class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, W, ((1,2,3), (1,2,3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)

        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(gy, W, b = None, stride=self.stride, pad=self.pad,
                        outsize=(x.shape[2], x.shape[3]))
        gW = Conv2dGradW(self)(x, gy)
        gb = None
        if b.data is None:
            gb = gy.sum(axis=(0,2,3))

        return gx, gW, gb

def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)

class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = stride
        self.pad = pad
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        Weight = W
        SH, SW = self.stide
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape

        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(Weight, x, (0,1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                        to_matrix=False)
        
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb

def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy
                
#####################
# simple_pooling 구현
#####################
def pooling_sample(x, kernel_size, stride=1, pad=0):
    '''
    pooling 수행 -> 최대값만 추출
    Args:
        x(Variable) : conv연산을 끝낸 결과값
        kernel_size(int or tuple) : max-pooling할 범위
        stride(int or tuple) : 움직이는 범위
        pad(int or tuple) : 패딩
    Returns:
        채널을 유지한 최대 풀링 결과 (stride만큼 감소)
    '''
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = par(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)  # axis=1기준으로 2차원이면 가로값들중 최대값
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y

class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))
        
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)

#####################
# im2col / col2im
#####################

class Im2col(Function):
    '''
    커널을 적용할 영역을 꺼내어 한줄로 형상을 바꾸어 2차원 텐서로 변환
    '''
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return gx

def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    '''
    데이터에서 커널을 적용할 영억을 꺼내어 한줄로 shape를 변경해 2차원 텐서로 변환
    Args:
        x(dezero.Variable of ndarray) : 변수 shape
        kernel_size(int or (int,int)) : 커널 size
        stride (int) : 보폭 크기
        pad (int or (int, int)) : 패딩 정도
        to_matrix(bool) : true이면 2차원 텐서를 다시 원본 shape으로 변환
    Return:
        to_matrix가 false이면 행렬곱이 이루어지지 않은 상태 (N, C, KH, KW, OH, OW)
        to_matrix가 true이면 행렬곱이 가능한 상태 (N*OH*OW, C*KH*KW)
    '''
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y

class Col2im(Function):
    '''
    im2col과 반대 역할
    '''
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,
        self.pad, self.to_matrix)
        return y
    
    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return gx

def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    '''
    im2col과 반대되는 역할
    '''
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

#########################
# numpy im2col / col2im
########################

def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    '''
    img를 2차원 텐서로 형태변환
    '''
    N, C, H, W = img.shape  # 배치 사이즈만큼
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    img = np.pad(img, ((0,0), (0,0), (PH, PH+SH-1),(PW, PW+SW-1)),
    mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:,:,j,i,:,:] = img[:,:,j:j_lim:SH,i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0,4,5,1,2,3)).reshape((N*OH*OW, -1))

    return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    
    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                    dtype=col.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        
    return img[:, :, PH:H + PH, PW:W + PW]