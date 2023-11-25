import numpy as np
import gymnasium as gym
from gymnasium import spaces
from objective import Objective
from numpy import linalg as LA

class GradDescentEnv_eval(gym.Env):
  def __init__(self,mode):
    M = float(1e5)

    self.mode = mode 
    self.function_nb = 0

    quadobj = Objective(self.function_nb,mode)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = 2./np.max(eigs)
    
    self.dimension = np.size(Q,1)

    # self.action_space = spaces.Box(low=0., high=max_step, shape=(1,), dtype=np.float64)
    self.action_space = spaces.Box(low=np.log10(1./10.), high=np.log10(2./10.), shape=(1,), dtype=np.float64)

    self.observation_space = spaces.Box(low=-M, high=M, shape=(int(2*self.dimension+1),), dtype=np.float64)

    self.iterations = 0

    self.tol = 1e-12

    self.max_iterations = 3e4

    self.nb_functions = 1

    self.ini = 5*np.ones((1,self.dimension))

  def step(self,action):
    action[0] = 10**action[0]
    pen = 0.
    if action[0]<1e-15:
      print('warning!eval')
      pen =  pen - 1e3
      action[0]=1e-15
    quadobj = Objective(self.function_nb,self.mode)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = 2./np.max(eigs)

    # Apply action
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


    # reward = (gamma**self.iterations)*(self.state[self.dimension] - new_func_val)
    reward = (self.state[self.dimension] - new_func_val) -self.iterations

    self.state[self.dimension] = new_func_val
    self.state[self.dimension+1:2*self.dimension+1] = jac_eval

    # print(self.function)
    # print(action[0])
    # print(self.state)
    
    # Terminal conditions

    if (self.iterations >= self.max_iterations):
      print("eval not within max iterations")
      if (self.function_nb == self.nb_functions - 1):
        terminate = True
        self.function_nb = 0
      else:
        self.function_nb += 1
        self.reset()
        terminate = False

    elif (LA.norm(self.state[0:self.dimension]) < self.tol):
      if (self.function_nb == self.nb_functions - 1):
        terminate = True
        self.function_nb = 0
      else:
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

    
    quadobj = Objective(self.function_nb,self.mode)
    func_val = quadobj.get_fval(self.ini)
    jac_eval = quadobj.get_jacval(self.ini)
    self.state = np.append(self.ini,func_val)
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
  
  