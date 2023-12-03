import numpy as np
import gymnasium as gym
from gymnasium import spaces
from objective import Objective
from numpy import linalg as LA
from gradient_descent import trunc_gradient_descent

class GradDescentEnv(gym.Env):
  def __init__(self,mode='train'):

    M = float(1e10)
    self.mode = mode
    quadobj = Objective(mode=self.mode)
    Q = quadobj.get_Q()
    
    self.dimension = np.size(Q,1)
    

    # self.action_space = spaces.Box(low=np.log10(1./10.), high=np.log10(2./1.), shape=(1,), dtype=np.float64)
    # self.action_space = spaces.Box(low=1./10., high=2./1., shape=(1,), dtype=np.float64)
    self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float64)

    self.observation_space = spaces.Box(low=-M, high=M, shape=(2*self.dimension,), dtype=np.float64)
    self.iterations = 0

    self.max_iterations = 5e3
    
    self.tol = 1e-8

    self.it = None

  def step(self,action):
    
    signs = self.state[self.dimension:2*self.dimension]
    self.state = 10**self.state[0:self.dimension]
    self.state = np.multiply(signs,self.state)

    #map interval [a,b] to [c,d]
    a = -1.
    b = 1.
    c = np.log10(1./10.)
    d = np.log10(2./1.)
    # c = 1./10.
    # d = 2./1.
    action[0] = c + (d - c)*(action[0] - a)/(b - a)
    action[0] = 10**action[0]

    pen = 0.
    if action[0]<1e-15:
      print('warning!')
      pen =  pen - 1e6
      action[0]=1e-15

    # Apply action
    quadobj = Objective(mode=self.mode)

    self.it = self.it - action[0]*self.state
    new_func_val = quadobj.get_fval(self.it)

    # Increase iterations
    self.iterations += 1
    
    # Calculate reward
    gamma = 0.99

    # reward = -np.log10(LA.norm(self.state[self.dimension+1:2*self.dimension+1])**2) + gamma*(-np.log10(LA.norm(jac_eval)**2)+np.log10(LA.norm(self.state[self.dimension+1:2*self.dimension+1])**2))
    # reward = (self.state[self.dimension] - new_func_val)**2 - self.iterations
    # reward = (gamma**self.iterations)*(self.state[self.dimension] - new_func_val)
    # reward = -np.log10(LA.norm(jac_eval)**2)
    if (new_func_val >= self.func_val) and (self.mode == 'train'):
      pen = pen - 1e6
    # reward = gamma*(0.9*new_func_val-self.state[self.dimension])

    self.func_val = new_func_val
    self.state = quadobj.get_jacval(self.it)
    self.state = self.state[0]
    
    # Terminal conditions

    if (self.iterations >= self.max_iterations):
      print("training not within max iterations")
      terminate = True

    elif (LA.norm(self.state) < self.tol):

      terminate = True

    else:
      terminate = False

    truncated = False
    if (LA.norm(self.state, np.inf) > 1e10):
    # if (LA.norm(self.state, np.inf) > 1e5):
      pen = pen -1e6
      truncated = True
    # Set placeholder for info
    info = {}

    if terminate:
      reward = 0.
    else:
      reward = -1.

    reward = reward + pen
    
    signs = np.sign(self.state)
    for i in range(self.dimension):
      if abs(self.state[i]) < 1e-15:
        self.state[i] = 1e-15
    self.state = np.log10(abs(self.state))
    self.state = np.append(self.state,signs)

    # Return step information
    
    action[0] = np.log10(action[0])
    return self.state, reward, terminate, truncated, info
  

  def get_iterate(self):
    return self.it
  
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    if self.mode=='train':
      ini = -5. + 10*np.random.rand(1,self.dimension)
    else:
      ini = 5*np.ones((1,self.dimension))

    quadobj = Objective(mode=self.mode)
    self.it = ini
    self.state = quadobj.get_jacval(ini)
    self.state = self.state[0]
    self.func_val = quadobj.get_fval(ini)
  
    signs = np.sign(self.state)
    for i in range(self.dimension):
      if abs(self.state[i]) < 1e-15:
        self.state[i] = 1e-15
    self.state = np.log10(abs(self.state))
    self.state = np.append(self.state,signs)

    self.iterations = 0
    info = {}

    return self.state, info

