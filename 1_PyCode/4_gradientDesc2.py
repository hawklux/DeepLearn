# Gradient Descent (경사하강법)
# f(x) = a*x1**2 + b*x2**2
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

# 예제 함수 정의
def function_2(x):
    return x[0]**2 + x[1]**2

# 초기값 설정
init_x2 = np.array([-3.0, 4.0])  # 시작 위치

# 경사하강법 실행 및 각 단계 결과 저장
result2 = gradient_descent(function_2, init_x=init_x2, lr=0.1, step_num=100)
print("최적화 결과2:", result2)  # (0, 0)의 근사치 나옴


# exit()
# x[0]**2 + x[1]**2 그래프 보기
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 예제 함수 정의
def function_2(x):
    return x[0]**2 + x[1]**2

# 그리드 생성
x0 = np.arange(-5.0, 5.0, 0.1)
x1 = np.arange(-5.0, 5.0, 0.1)
X0, X1 = np.meshgrid(x0, x1)

# Z 값 계산
Z = X0**2 + X1**2

# 3D 그래프 그리기
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X0, X1, Z, cmap="viridis", edgecolor='none', alpha=0.7)
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
ax.set_title('3D plot of function_2 (x[0]**2 + x[1]**2)')
plt.show()
