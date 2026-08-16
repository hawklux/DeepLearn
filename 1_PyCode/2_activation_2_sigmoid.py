# Sigmoid

import numpy as np
import matplotlib.pylab as plt

# Sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step function
def step_funct(x):
    y = x > 0
    return y.astype(np.int32)

# Numpy 배열로 테스트해보기
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))
x1 = 1.0 + x
print(sigmoid(x1))
x2 = 1.0 / x
print(sigmoid(x2))

# Sigmoid 그래프 그리기
x = np.arange(-5.0, 5.0, 0.1)  # -5~5, 0.1씩
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# Sigmoid + Step function 그래프
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_funct(x)
plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)
plt.show()
