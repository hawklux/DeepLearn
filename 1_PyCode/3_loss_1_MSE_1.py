# Mean Squared Error

import numpy as np
import matplotlib.pylab as plt

# 정답 (one-hot encoding으로 하나가 선별됨)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
# 예측값 
y = [0.05, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# 평균제곱오차 (예측값 - 정답)
def mse(y, t):
    return 0.5 * np.sum((y-t)**2)

res = mse(np.array(y), np.array(t))
print(res)    # 0.09375

