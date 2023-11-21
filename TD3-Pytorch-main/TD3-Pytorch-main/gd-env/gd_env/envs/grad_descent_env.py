import numpy as np
import gymnasium as gym
from gymnasium import spaces
from objective import Objective
from numpy import linalg as LA

class GradDescentEnv(gym.Env):
  def __init__(self, mode, function_nb=0):
    M = float(1e5)

    self.function_nb = function_nb

    self.mode = mode

    quadobj = Objective(self.function_nb, self.mode)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = float(2./np.max(eigs))
    # np.savetxt("max_step.txt", np.array([max_step]), fmt='%4.15f', delimiter=' ') 
    
    self.dimension = np.size(Q,1)

    # self.action_space = spaces.Box(low=0., high=0.99*max_step, shape=(1,), dtype=np.float64)
    self.action_space = spaces.Box(low=0., high=2, shape=(1,), dtype=np.float64)

    self.observation_space = spaces.Box(low=-M, high=M, shape=(int(2*self.dimension+1),), dtype=np.float64)

    self.iterations = 0

    self.nb_passes = 0

    self.nb_train_func = 1

    self.nb_eval_func = 1

    self.max_iterations = 500
    
    self.tol = 1e-12
    

  def step(self,action):
    pen = 0.
    if action[0]<1e-15:
      print('warning!')
      pen =  pen - 1e6
      action[0]=1e-15
    # Apply action
    quadobj = Objective(self.function_nb, self.mode)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = float(2./np.max(eigs))
    if action[0] > max_step:
      pen = pen - 1e6
      action[0] = max_step

    self.state[0:self.dimension] = self.state[0:self.dimension] - action[0]*self.state[self.dimension+1:2*self.dimension+1]
    new_func_val = quadobj.get_fval(self.state[0:self.dimension])
    jac_eval = quadobj.get_jacval(self.state[0:self.dimension])

    # Increase iterations
    self.iterations += 1

    # Calculate reward
    gamma = 0.5

    reward = (self.state[self.dimension] - new_func_val) -self.iterations
    # reward = (gamma**self.iterations)*(self.state[self.dimension] - new_func_val)

    self.state[self.dimension] = new_func_val
    self.state[self.dimension+1:2*self.dimension+1] = jac_eval
    
    # Terminal conditions
    if self.mode == 'train':
      nb_functions = self.nb_train_func
    elif self.mode == 'eval':
      nb_functions = self.nb_eval_func
    else:
      nb_functions = 1

    if (self.iterations >= self.max_iterations):
      print("training not within max iterations")
      if (self.function_nb >= nb_functions - 1):
        terminate = True
        self.function_nb = 0
        self.nb_passes += 1
      else:
        # print(self.function_nb)
        self.function_nb += 1
        self.reset()
        terminate = False

    elif (LA.norm(self.state[0:self.dimension]) < self.tol):
      if (self.function_nb >= nb_functions - 1):
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
    # if terminate:
    #   reward = 0.
    # else:
    #   reward = -1.
    reward = reward + pen
    # Return step information
    return self.state, reward, terminate, False, info
  
  def render(self):
    pass

  def get_function_nb(self):
    return self.function_nb

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    if self.mode == 'train':
      ini = -5. + 10*np.random.rand(1,self.dimension)
    else:
      ini = 5*np.ones((1,self.dimension))
    quadobj = Objective(self.function_nb, self.mode)
    func_val = quadobj.get_fval(ini)
    jac_eval = quadobj.get_jacval(ini)
    self.state = np.append(ini,func_val)
    self.state = np.append(self.state,jac_eval)
    self.iterations = 0

    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    
    max_step = 2./np.max(eigs)
    # np.savetxt("max_step.txt", np.array([max_step]), fmt='%4.15f', delimiter=' ')
    self.action_space = spaces.Box(low=0., high=0.99*max_step, shape=(1,), dtype=np.float64)
    M = float(1e5)
    self.observation_space = spaces.Box(low=-M, high=M, shape=(int(2*self.dimension+1),), dtype=np.float64)
    info = {}

    return self.state, info
  