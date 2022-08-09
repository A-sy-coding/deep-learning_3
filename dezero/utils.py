# utils.py 파일에는 계산 그래프를 시각화 하는 함수를 구현하려고 한다.
import os
import subprocess
import numpy as np
import urllib.request

def _dot_var(v, verbose=False):  # 함수 앞에 _가 들어가면 로컬에서만 사용한다는 의미이다.
    dot_var = '{} [label="{}" , color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

# dezero 함수를 DOT 언어로 변환하는 함수 구현
def _dot_func(f):
    dot_func = '{} [label={}, color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)  # 작동하는 클래스 이름을 가져와서 label로 설정한다.

    dot_edge = '{} -> {}\n'  # 노드들을 연결해주는 선 정의
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))

    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)  # func에 값 추가
    txt += _dot_var(output,verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

# dot 언어를 작성하고 해당 dot 명령실행까지 한번에 해주는 함수를 구현
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)  # dot graph 언어 생성

    # dot 데이터를 파일에 저장
    tmp_dir = os.path.join(os.getcwd(), '.dezero')  # 임시 경로 설정
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:  # 파일 저장
        f.write(dot_graph)

    # dot 명령 호출
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

################################
# utility functions for numpy
################################

# 행렬을 주어진 shape으로 값을 변환하여 출력한다. 즉, x의 원소의 합을 구해 shape형상으로 만들어주는 함수
def sum_to(x, shape):

    ndim = len(shape)  # 차원수
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i,sx in enumerate(shape) if sx==1])
    # print(lead_axis, axis)
    y = x.sum(lead_axis + axis, keepdims=True)

    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

    def reshape_sum_backward(gy, x_shape, axis, keepdims):
        """Reshape gradient appropriately for dezero.functions.sum's backward.
        Args:
            gy (dezero.Variable): Gradient variable from the output by backprop.
            x_shape (tuple): Shape used at sum function's forward.
            axis (None or int or tuple of ints): Axis used at sum function's
                forward.
            keepdims (bool): Keepdims used at sum function's forward.
        Returns:
            dezero.Variable: Gradient variable which is reshaped appropriately
        """
        ndim = len(x_shape)
        tupled_axis = axis
        if axis is None:
            tupled_axis = None
        elif not isinstance(axis, tuple):
            tupled_axis = (axis,)

        if not (ndim == 0 or tupled_axis is None or keepdims):
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
            shape = list(gy.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)
        else:
            shape = gy.shape

        gy = gy.reshape(shape)  # reshape
        return gy

# logsumexp --> log계산할 때 0이면 오류가 발생하므로 해당 오류를 방지하기 위해 x_min과 x_max를 설정하여 그 값들 범위 밖으로 벗어나지 않도록 한다.
def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m

############################
# file process
############################

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.
    The file at the `url` is downloaded to the `~/.dezero`.
    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.
    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path

# im2col & col2im에서 사용
def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2*p