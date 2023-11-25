from benderopt import minimize
from gradient_descent import gradient_descent
from objective import Objective
import numpy as np
import rbfopt
import numpy as np
import os
import logging
from numpy import linalg as LA
os.environ['LD_LIBRARY_PATH']='/volume1/scratch/rwang/pieter/lennert-evens/dist/lib'
# def black_box_opt(X:np.ndarray, function_nb:int) -> {np.ndarray, np.ndarray, np.ndarray, int}:

#     #logging.basicConfig(level=logging.INFO) # logging.INFO will print less information
#     quadobj = Objective(function_nb)
#     Q = quadobj.get_Q()
#     eigs, _ = LA.eig(Q)
#     max_step = float(2./np.max(eigs))
#     dimension = np.size(Q,1)

#     def reward_func(traj,iter):
#         func_eval = np.zeros((np.size(traj,0),1))
#         for i in range(np.size(traj,0)):
#             func_eval[i,:] = quadobj.get_fval(traj[i,:])
#         reward = 0.
#         gamma = 0.5
#         for i in range(np.size(traj,0)-1):
#             reward += (gamma**(i+1))*(func_eval[i,:] - func_eval[i+1,:]) - np.size(traj,0)
#         return reward

#     def obj_funct(stepsize):
#         traj, iter, _ = gradient_descent(X,0,False,stepsize)
#         reward = reward_func(traj,iter)
#         return -reward

#     # We define the parameters we want to optimize:
#     optimization_problem_parameters = [
#         {   "name": "stepsize", 
#             "category": "uniform",
#             "search_space": {"low": 0., "high": max_step,}}]

#     terminate = False
#     tol = 1e-12
#     max_iter = 1e4
#     step_cache = []
#     iter = 0
#     traj = X
#     nb_fe = 0
#     fe_cache = np.array([nb_fe])
#     while terminate is False:

#         best_sample = minimize(obj_funct, optimization_problem_parameters, number_of_evaluation=50)
#         opt_step = best_sample["stepsize"]
#         step_cache = np.append(step_cache,opt_step)
#         grad = quadobj.get_jacval(X)
#         X = gradient_descent_step(X,grad,opt_step)
#         nb_fe += dimension
#         traj = np.append(traj,X,axis=0)
#         fe_cache = np.append(fe_cache,nb_fe)
#         iter += 1
#         if (LA.norm(X) < tol) or (iter == max_iter):
#             terminate = True


#     return traj, step_cache, fe_cache, iter

def black_box_opt(X:np.ndarray, quadobj:Objective) -> {np.ndarray, np.ndarray, np.ndarray, int}:


    def reward_func(traj,iter):
        func_eval = np.zeros((np.size(traj,0),1))
        for i in range(np.size(traj,0)):
            func_eval[i,:] = quadobj.get_fval(traj[i,:])
        reward = 0.
        gamma = 0.9
        for i in range(np.size(traj,0)-1):
            # reward += (gamma**(i+1))*(func_eval[i,:] - func_eval[i+1,:])
            reward += (func_eval[i,:] - func_eval[i+1,:]) - np.size(traj,0)
        return reward

    def obj_funct(stepsize):
        traj, iterations, fe_cache, _ = gradient_descent(X,0,False,stepsize)
        file = open("bbo_it.txt", "r")
        line = file.readlines()
        total_fe = float(np.fromstring(line[0], dtype=float, sep=' '))
        np.savetxt('bbo_it.txt', np.array([total_fe+fe_cache[-1]]), fmt='%4.15f')
        reward = reward_func(traj,iterations)
        return iterations

    # We define the parameters we want to optimize:
    optimization_problem_parameters = [
        {   "name": "stepsize", 
            "category": "uniform",
            "search_space": {"low": 0., "high": quadobj.get_max_step(),}}]

    best_sample = minimize(obj_funct, optimization_problem_parameters, number_of_evaluation=50)
    opt_step = best_sample["stepsize"]

    # bb = rbfopt.RbfoptUserBlackBox(1, np.array([0.]), np.array([quadobj.get_max_step()]),
    #                            np.array(['R']), obj_funct)
    # settings = rbfopt.RbfoptSettings(max_evaluations=50, minlp_solver_path='/volume1/scratch/rwang/pieter/lennert-evens/dist/bin/bonmin')
    # alg = rbfopt.RbfoptAlgorithm(settings, bb)
    # opt_step, x, itercount, evalcount, fast_evalcount = alg.optimize()

    return opt_step