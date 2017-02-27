import pandas as pd
from numpy import *
from numpy.linalg import norm

## The data is referred to from tutorial code in week 1-2: gradient_descent.py
dat = pd.read_csv("galaxy.data")
x1 = dat.loc[:,"east.west"].as_matrix()
x2 = dat.loc[:, "north.south"].as_matrix()
y = dat.loc[:, "velocity"].as_matrix()

def g(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    # 
    # term1 = sum(dot(theta.T, x))
    # term2 = term1.T - y 
    # return term2 ** 2
    return sum(sum((dot(theta.T, x).T - y)**2))

    
def dg(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    # 
    # print "IN DG()"
    # print "x", shape(x)
    # print "y", shape(y)
    # print "theta", shape(theta)

    term = dot(theta.T, x) 
    term2 = term.T - y    
    term3 = dot(x, term2)
    
    return 2 * dot(x, (dot(theta.T, x) - y))


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 50000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        print "shape(x,y,t)", df(x,y,t)
        
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

x = vstack((x1, x2))
theta = array([-3,3,2])
h = 1e-5
finite_diff = (g(x, y, theta+array([0,h,0])) - g(x, y, theta-array([0,h,0]))) / (2*h)
predicted = dg(x, y, theta)
print "====================================================="
print "FINITE DIFFERENCE\n"
print finite_diff
print "====================================================="
print "DG()"
print predicted[1]

x = vstack((x1, x2))
theta0 = array([0., 0., 0.])