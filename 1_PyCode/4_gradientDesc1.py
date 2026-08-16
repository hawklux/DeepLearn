# Gradient Descent (경사하강법)
# f(x) = a*x**2 + b*x + c
import numpy as np
import matplotlib.pylab as plt

# 기울기 구하기 함수 (미분)
def numerical_gradient(f, x):
    h = 1e-4 #0.0001         # 아주 작은 변동값
    grad = np.zeros_like(x)  # x와 shape 같은 배열 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h  # x 미소 변화
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h  # x 미소 변화
        fxh2 = f(x)
        
        grad[idx] = (fxh1 -fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원
    
    return grad    

# 경사 하강법 정의 (대상함수, 시작점, 학습률, 반복횟수)
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# 예제 함수 정의 (예: f(x) = ax^2 + bx + c)
def function_1(x):
    a, b, c = 1, -2, 1
    return a*x[0]**2 + b*x[0] + c

# 초기값 설정
init_x1 = np.array([-3.0])  # 시작 위치

# 경사하강법 실행 및 각 단계 결과 저장
result1 = gradient_descent(function_1, init_x=init_x1, lr=0.1, step_num=100)
print("최적화 결과2:", result1)  # (1, 0)의 근사치 나옴


# exit()
# 예제 함수 그래프 보기
import numpy as np
import matplotlib.pyplot as plt

# 함수 정의 
def function_1(x):
    a, b, c = 1, -2, 1
    return a*x**2 + b*x + c

# x 값의 범위를 설정하고 함수값 계산
x_vals = np.arange(-5.0, 5.0, 0.1)
y_vals = function_1(x_vals)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="f(x) = x^2 -2x + 1")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Graph of fuction_1")
plt.legend()
plt.grid(True)
plt.show()