# XOR 게이트

import numpy as np
inputs = [(0,0), (0,1), (1,0), (1,1)]

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.51
    y = np.sum(w*x) - b
    if y <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.1
    y = np.sum(w*x) + b
    if y <= b:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.51
    y = np.sum(w*x) - b
    if y <= 0:
        return 1
    else:
        return 0

def XOR(x1, x2):
    y_NAND = NAND(x1, x2)
    y_OR = OR(x1, x2)
    y_XOR = AND(y_NAND, y_OR)
    return(y_XOR)

for input_ in inputs:
    x1, x2 = input_
    y_OR = XOR(x1, x2)
    print(y_OR)

