import numpy as np
import rbfopt
import gradient_descent

X0 = 5*np.ones((2,1))
def obj_funct(stepsize):
  _, iter, _ = gradient_descent(X0,1,False,stepsize)
  reward = -iter
  return -reward

bb = rbfopt.RbfoptUserBlackBox(1, 0.,2./5.536611758572813,np.array(['R']), obj_funct)
settings = rbfopt.RbfoptSettings(max_evaluations=50, minlp_solver_path='../../../Bonmin',nlp_solver_path='../../../Ipopt')
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
