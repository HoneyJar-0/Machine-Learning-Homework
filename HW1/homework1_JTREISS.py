import numpy as np

##############################
# Part 1
##############################
# Write your answers to the math problems below. To represent exponents, use the ^ symbol.
# 1. y + z + 2yz^2     partial der. w/respect to x
#    x + 2xz^2         partial der. w/respect to y
#    x + 4xyz          partial der. w/respect to z
#
# 2. ln(v)                  partial der. w/respect to u
#    u/v + 2v * e^(v^2)     partial der. w/respect to v
#
# 3. B is 4x6
#
# 4. partial derivative of g(x,y) with respect to x: -1
#    partial derivative of g(x,y) with respect to y: -2
#

##############################
# Part 2
##############################

def problem1 (A, B, C):
    sizeA = A.size()
    sizeB = B.size()

    if(len(sizeA) > 1 and len(sizeB) > 1):
        if((sizeA[0] != sizeB[1]) and (sizeA[1] == sizeB[1])):
            A = np.transpose(A)
        else:
            print("Incompatible shapes between A and B")
            return None

    dotted = np.dot(A, B)

    if(dotted.size() == C.size()):
        return np.subtract(dotted, C) 
    else:
        print("Incompatible shapes between AB and C")
        return None

def problem2 (A):
    return np.ones(A)

def problem3 (A):
    return np.fill_diagonal(A, 0)

def problem4 (A, i):
    return np.sum(A, i)

def problem5 (A, c, d):
    return np.mean(A[A >= c and A <= d])

def problem6 (A, k):
    w, v = np.linalg.eig(A) 
    w = np.flip(np.argsort(np.abs(w)))[:k]
    return v[:, w]

def problem7 (A, x):
    return np.linalg.solve(A, x)

def problem8 (x, k):
    x = np.asarray(x)
    x = x[np.newaxis, :]
    return np.repeat(x, k, axis = 0)

def problem9 (A):
    return np.random.permutation(A)

def problem10 (A):
    return np.mean(A, axis = 1)