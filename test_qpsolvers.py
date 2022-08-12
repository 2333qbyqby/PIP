from numpy import array, dot
from qpsolvers import solve_qp

M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # this is a positive definite matrix
q = dot(array([3., 2., 3.]), M)
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.])
A = array([1., 1., 1.])
b = array([1.])

x = solve_qp(P, q, G, h, A, b, solver="osqp")
print("QP solution: x = {}".format(x))
