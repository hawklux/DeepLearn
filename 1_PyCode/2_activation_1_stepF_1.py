# Step Function (계단 함수)

import numpy as np               # pip install numpy
import matplotlib.pylab as plt   # pip install matplotlib

# 수를 받는 Step function
def step_funct(x):
    if x > 0:
        return 1
    else:
        return 0

# Numpy 배열을 받는 Step funcion
def step_funct_np(x):
    y = x > 0                   # true, false
    return y.astype(np.int32)   # true=1, false=0

# Numpy 배열을 넣어 테스트 해보기
x = np.array([-1, 2, -3, 4])
print(step_funct_np(x))         #[0 1 0 1]

# Step function 그래프 그리기
x = np.arange(-5.0, 5.0, 0.1)
y = step_funct_np(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
