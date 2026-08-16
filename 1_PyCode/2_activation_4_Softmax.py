# Softmax로 출력층 설계하기
import numpy as np
import matplotlib.pylab as plt

a1 = np.array([0.3, 2.9, 4.0])

# 1. Softmax 기본 함수 구현
def softmax_1(a1):
    exp_a = np.exp(a1)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 2. Softmax 보완 함수 구현
# 지수함수 특성상 입력값이 100만 되도 10**40이 넘는 큰 수가 되어 무한으로 수렴함
# 이러한 Overflow를 해결하기 위해 "입력값(a) - 최대입력값(c)"를 해줌
a2 = np.array([1010, 1000, 900])
def softmax_2(a2):
    c = np.max(a2)
    exp_a = np.exp(a2 - c)  # Overflow 방지 ([0, -10, -20])
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#########################
# 참고: 지수함수 확인
exp_a1 = np.exp(a1)
print(exp_a1)

sum_exp_a1 = np.sum(exp_a1)
print(sum_exp_a1)
#########################
exp_a2 = np.exp(a2)
print(exp_a2)  # [inf inf inf]

sum_exp_a2 = np.sum(exp_a2)
print(sum_exp_a2)  # inf

y3 = exp_a2 / sum_exp_a2
print(y3)   # array([nan, nan, nan])  무한대로 수렴해서.
#########################

# 3. 그래프로 보기
x = np.arange(-5.0, 5.0, 0.1)
y1 = softmax_1(x)
y2 = softmax_2(x)
plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.show()
