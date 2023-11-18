import numpy as np
import gymnasium as gym
from gymnasium import spaces
from objective import Objective
from numpy import linalg as LA

class GradDescentEnv_eval(gym.Env):
  def __init__(self):
    M = float(1e5)

    self.function_nb = 0

    quadobj = Objective(self.function_nb,'eval')
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = 2./np.max(eigs)
    # np.savetxt("max_step.txt", np.array([max_step]), fmt='%4.15f', delimiter=' ') 
    
    self.dimension = np.size(Q,1)

    self.action_space = spaces.Box(low=0., high=max_step, shape=(1,), dtype=np.float64)

    self.observation_space = spaces.Box(low=-M, high=M, shape=(int(2*self.dimension+1),), dtype=np.float64)

    self.iterations = 0

    self.nb_passes = 0
    

  def step(self,action):
    pen = 0.
    if action[0]<1e-15:
      print('warning!eval')
      pen =  pen - 1e3
      action[0]=1e-15
    # Apply action
    quadobj = Objective(self.function_nb,'eval')
    self.state[0:self.dimension] = self.state[0:self.dimension] - action[0]*self.state[self.dimension+1:2*self.dimension+1]
    new_func_val = quadobj.get_fval(self.state[0:self.dimension])
    jac_eval = quadobj.get_jacval(self.state[0:self.dimension])

    # Increase iterations
    self.iterations += 1

    # Calculate reward
    gamma = 0.9

    reward = (self.state[self.dimension] - new_func_val)**2 -self.iterations

    self.state[self.dimension] = new_func_val
    self.state[self.dimension+1:2*self.dimension+1] = jac_eval

    # print(self.function)
    # print(action[0])
    # print(self.state)
    
    # Terminal conditions
    max_iterations = 3e4
    tol = 1e-12
    nb_functions = 1
    if (self.iterations >= max_iterations):
      print("eval not within max iterations")
      if (self.function_nb == nb_functions - 1):
        terminate = True
        self.function_nb = 0
        self.nb_passes += 1
      else:
        # print(self.function_nb)
        self.function_nb += 1
        self.reset()
        terminate = False

    elif (LA.norm(self.state[0:self.dimension]) < tol):
      if (self.function_nb == nb_functions - 1):
        terminate = True
        self.function_nb = 0
        self.nb_passes += 1
      else:
        print(self.function_nb)
        self.function_nb += 1
        self.reset()
        terminate = False
    else:
      terminate = False

    # Set placeholder for info
    info = {}
    if terminate:
      reward = 0.
    else:
      reward = -1.
    reward = reward + pen
    # Return step information
    return self.state, reward, terminate, False, info
  
  def render(self):
    pass

  def get_function_nb(self):
    return self.function_nb

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    ini = -5. + 10*np.random.rand(1,self.dimension)
    ini = 5*np.ones((1,self.dimension))
    quadobj = Objective(self.function_nb,'eval')
    func_val = quadobj.get_fval(ini)
    jac_eval = quadobj.get_jacval(ini)
    self.state = np.append(ini,func_val)
    self.state = np.append(self.state,jac_eval)
    self.iterations = 0
    
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    
    max_step = 2./np.max(eigs)
    # np.savetxt("max_step.txt", np.array([max_step]), fmt='%4.15f', delimiter=' ')
    self.action_space = spaces.Box(low=0., high=max_step, shape=(1,), dtype=np.float64)
    M = float(1e5)
    self.observation_space = spaces.Box(low=-M, high=M, shape=(int(2*self.dimension+1),), dtype=np.float64)
    info = {}

    return self.state, info
  
  