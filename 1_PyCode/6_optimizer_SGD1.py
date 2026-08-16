# 버전 1: 간단한 SGD optimizer 흉내
# https://app.notion.com/p/Optimizer-3a1a5c22d8918010a4e9e1dade65d3f2?source=copy_link#3a1a5c22d891807e8d2fe5abffaeae2e
def sgd(param, grad, lr=0.1):
    return param - lr*grad

#예시
w = 10.0
grad = 2.5  # 손실률 w로 미분한 값이라고 가정함.
learning_rate = 0.1

new_w = sgd(w, grad, learning_rate)

print("업데이트 전 w:", w)
print("grad:", grad)
print("업데이트 후 w:", new_w)