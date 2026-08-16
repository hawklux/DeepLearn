# OR 게이트

import numpy as np

inputs = [(0,0), (1,0), (0,1), (1,1)]

def OR_(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.5
    y = np.sum(x*w) + b

    if y <= b:
        return 0
    else:
        return 1
    
for input_ in inputs:
    x1, x2 = input_
    res = OR_(x1, x2)
    print(res)
