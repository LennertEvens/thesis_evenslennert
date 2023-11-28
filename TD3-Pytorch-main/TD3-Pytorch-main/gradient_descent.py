from objective import Objective
import numpy as np
from numpy import linalg as LA

def gradient_descent_step(X:np.ndarray, grad:np.ndarray, step:float) -> np.ndarray:
    X = X - step*grad
    return X

def trunc_gradient_descent(X:np.ndarray, grad:np.ndarray, step:float, iterations: int) -> np.ndarray:
    for i in range(iterations-1):
        X = gradient_descent_step(X,grad,step)
    return X

def exact_linesearch(X:np.ndarray, quadobj:Objective) -> float:
    grad = quadobj.get_jacval(X)
    Q = quadobj.get_Q()
    return np.matmul(grad,np.transpose(grad))/(np.matmul(np.matmul(grad,Q),np.transpose(grad)))

def gradient_descent(X, function_nb, linesearch=False, stepsize=None, bbo=None):
    tol = 1e-12
    max_iter = 1e4
    terminate = False
    traj = X
    quadobj = Objective(function_nb,'test')
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    gamma = 0.50
    eigs = np.real(eigs)
    max_step = 2./np.max(eigs)
    step = gamma*max_step
    iter = 0
    dimension = np.size(Q,1)

    if stepsize is not None:
        step = stepsize
    # linesearch parameters
    beta = 0.9
    gamma = 0.1
    nb_fe = 0
    fe_cache = np.array([nb_fe])
    step_cache = []
    while terminate is False:
        grad = quadobj.get_jacval(X)
        if linesearch:
            # Backtracking linesearch
            # t = 1.
            # grad = quadobj.get_jacval(X)
            # nb_fe += dimension
            # while quadobj.get_fval(X-t*grad) >= (quadobj.get_fval(X) - gamma*t*(LA.norm(grad)**2)):
            #     nb_fe += 2
            #     t = beta*t
            #     if t<1e-15:
            #         print("no LS solution")
            #         break
            # step = t

            # exact linesearch
            step = exact_linesearch(X,quadobj)

        if bbo:
            from bbo import black_box_opt
            step = black_box_opt(X,quadobj)
        
        step_cache = np.append(step_cache,step)
        X = gradient_descent_step(X,grad,step)
        traj = np.append(traj,X,axis=0)
        iter += 1
        nb_fe += 1
        fe_cache = np.append(fe_cache,nb_fe)
        if (LA.norm(X) < tol) or (iter == max_iter):
            terminate = True
    
    return traj, iter, fe_cache, step_cache



    