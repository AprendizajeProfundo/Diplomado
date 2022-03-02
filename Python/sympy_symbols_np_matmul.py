import sympy
import numpy as np
from sympy.abc import x
from sympy import MatrixSymbol, Matrix

#expr = 1+sympy.sin(x)**2/sympy.cos(x)**2

X = MatrixSymbol('X', 3, 3)
Xt = np.transpose(X)

Ones = np.matrix([[1],[1],[1]])
Onest = np.matrix([1,1,1])
I = np.identity(3)
n = sympy.symbols("n")

M = (I-(1/n)*np.matmul(Ones,Onest))
S = (1/n)*np.matmul(np.matmul(Xt,M), X)
S[0,0]
