# Learning rate (학습률): 앞선 경사하강법 코드를 활용함.
import numpy as np
import matplotlib.pyplot as plt

# 1차 함수 정의: f(x) = x**2
def funct_1(x):
    return x**2

# 기울기 계산 함수
def numerical_gradient(f, x):
    h = 1e-4  # 아주 작은 변동값
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)   # f(x + h)

        x[idx] = tmp_val - h
        fxh2 = f(x)   # f(x - h)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

# 경사하강법 함수
def gradient_descent(f, init_x, lr, step_num=20):
    x = init_x
    x_history = [x.copy()]  # 그래프 그리기 위한 변동 기록
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        x_history.append(x.copy())
        
    return np.array(x_history)

# 초기값 설정
init_x = np.array([-10.0])  # 시작 위치

# 다양한 학습률 설정
learning_rates = [0.1, 0.01, 0.001]

# 그래프 그리기
plt.figure(figsize=(10, 6))
x_vals = np.linspace(-10, 10, 100)
y_vals = funct_1(x_vals)

# 함수 그래프
plt.plot(x_vals, y_vals, "k-", label="f(x) = x^2")

# 학습률별 경사 하강법 경로
for lr in learning_rates:
    history = gradient_descent(funct_1, init_x.copy(), lr, step_num=20)
    plt.plot(history, funct_1(history), 'o-', label=f"Learing rate = {lr}")
    
plt.xlabel("X")
plt.ylabel("f(x)")
plt.legend()
plt.title("Grad w/ different learning rage")
plt.grid(True)
plt.show()