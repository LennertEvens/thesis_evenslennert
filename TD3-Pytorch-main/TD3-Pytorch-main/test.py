# import rbfopt
import numpy as np
from numpy import linalg as LA
# import os
# os.environ['LD_LIBRARY_PATH']='/volume1/scratch/rwang/pieter/lennert-evens/dist/lib'
# def obj_funct(x):
#   return x[0]*x[1] - x[2]

# bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
#                                np.array(['R', 'I', 'R']), obj_funct)
# settings = rbfopt.RbfoptSettings(max_evaluations=50, minlp_solver_path='/volume1/scratch/rwang/pieter/lennert-evens/dist/bin/bonmin')
# alg = rbfopt.RbfoptAlgorithm(settings, bb)
# val, x, itercount, evalcount, fast_evalcount = alg.optimize()
eig = 1 + (10-1)*np.random.rand(10,1)
Q_all = np.diagflat(eig)
Q_all = np.reshape(Q_all,(1,int(np.size(Q_all)/1)))
np.savetxt("test_func.txt", Q_all, fmt='%4.15f', delimiter=' ') 