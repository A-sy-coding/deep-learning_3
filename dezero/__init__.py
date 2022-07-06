# __init__.py파일은 모듈을 import할 때 가장 먼저 실행되는 파일이다.

# core_simple.py 파일과 core.py 파일 중 하나를 선택해 import 하도록 설정한다.

# is_simple_core = True  # core_simple.py 파일을 사용할 것인지에 관한 여부
is_simple_core = False  # core.py파일을 사용

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable

setup_variable()   # 오버로드한 연산자들 사전 정의