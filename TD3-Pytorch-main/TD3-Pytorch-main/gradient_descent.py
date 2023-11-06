from objective import Objective
import numpy as np
from numpy import linalg as LA

def gradient_descent(X, function_nb, linesearch=False):
    tol = 1e-12
    max_iter = 1e3
    terminate = False
    traj = X
    quadobj = Objective(function_nb)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    gamma = 0.50
    eigs = np.real(eigs)
    max_step = 2./np.max(eigs)
    step = gamma*max_step
    iter = 0

    # linesearch parameters
    beta = 0.9
    gamma = 0.1
    nb_fe = 0
    fe_cache = np.array([nb_fe])
    while terminate is False:

        if linesearch:
            t = 1.
            grad = quadobj.get_jacval(X)
            nb_fe += 2
            while quadobj.get_fval(X-t*grad) >= (quadobj.get_fval(X) - gamma*t*(LA.norm(grad)**2)):
                nb_fe += 2
                t = beta*t
                if t<1e-15:
                    print("no LS solution")
                    break
            step = t

        X = X - step*quadobj.get_jacval(X)
        traj = np.append(traj,X,axis=0)
        iter += 1
        nb_fe += 2
        fe_cache = np.append(fe_cache,nb_fe)
        if (LA.norm(X) < tol) or (iter == max_iter):
            terminate = True
    
    return traj, iter, fe_cache


    