import rbfopt
import numpy as np
import os
os.environ['LD_LIBRARY_PATH']='/volume1/scratch/rwang/pieter/lennert-evens/dist/lib'
def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), obj_funct)
settings = rbfopt.RbfoptSettings(max_evaluations=50, minlp_solver_path='/volume1/scratch/rwang/pieter/lennert-evens/dist/bin/bonmin')
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()