# Sigmoid를 퍼셉트론에 적용하기

import numpy as np

# Sigmoid 함수 정의 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Perceptron 함수 정의
def perceptron(x, w, b):
    z = np.dot(x, w) + b
    return sigmoid(z)

# 입력 데이터 만들기
x = np.array([1.5, 2.0])

# 가중치와 편향 초기화
w = np.array([0.4, 0.6])
b = -0.3

# 퍼셉트론 출력
y = perceptron(x, w, b)
print(f"Perceptron Output: {y:.2f}")
