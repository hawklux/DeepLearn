# ReLU
import numpy as np
import matplotlib.pylab as plt

# ReLU 함수 구현
def relu(x):
    return np.maximum(0, x)

# numpy 배열로 테스트 해보기
x = np.array([-1.0, 1.0, 2.0])
print(relu(x))

# 그래프로 그려보기
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
