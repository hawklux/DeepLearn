# NAND 게이트
import numpy as np

inputs = [(0,0), (1,0), (0,1), (1,1)]

def NAND (x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.51
    y = np.sum(w*x) + b
    
    if y <= 0:
        return 0
    else:
        return 1
    
for input_ in inputs:
    x1, x2 = input_
    y = NAND(x1, x2)
    print(y)
    