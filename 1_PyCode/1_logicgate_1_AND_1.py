inputs = [(0,0), (1,0), (0,1), (1,1)]

# AND 게이트 기본
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    y = w1*x1 + w2*x2
    if y <= theta:
        return 0
    elif y > theta:
        return 1


# AND 게이트 numpy 계산
import numpy as np

def AND_np(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.51   # 0.5 초과
    y = np.sum(w*x) + b
    if y <= 0:
        return 0
    else:
        return 1

print("AND 실행")
for input_ in inputs:
    x1, x2 = input_
    y = AND(x1, x2)
    print(y)

print("AND_np 실행")
for input_ in inputs:
    x1, x2 = input_
    y = AND_np(x1, x2)
    print(y)
    