import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to

#####################
# conv2_simple 구현
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