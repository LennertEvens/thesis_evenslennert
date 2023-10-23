import numpy as np
from gym import Env
from gym.spaces import Box
from objective import Objective
from numpy import linalg as LA

class GradDescentEnv(Env):
  def __init__(self):
    M = float(1e5)
    print(test)
    self.function_nb = 0

    quadobj = Objective(self.function_nb)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = 2./np.max(eigs)

    self.action_space = Box(low=0., high=max_step, shape=(1,), dtype=np.float64)

    self.observation_space = Box(low=-M, high=M, shape=(5,), dtype=np.float64)

    self.iterations = 0

    self.nb_passes = 0
    

  def step(self,action):

    # Apply action
    quadobj = Objective(self.function_nb)
    self.state[0:2] = self.state[0:2] - action[0]*self.state[3:5]
    new_func_val = quadobj.get_fval(self.state[0:2])
    jac_eval = quadobj.get_jacval(self.state[0:2])

    # Increase iterations
    self.iterations += 1

    # Calculate reward
    gamma = 0.5
  
    reward = (self.state[2] - new_func_val)**2 - self.iterations
    self.state[2] = new_func_val
    self.state[3:5] = jac_eval

    # print(self.function)
    # print(action[0])
    # print(self.state)
    
    # Terminal conditions
    max_iterations = 1e4
    tol = 1e-12
    nb_functions = 10
    if (self.iterations >= max_iterations):
      print("Not within max iterations")
      self.function_nb += 1
      print(self.function_nb)
      self.reset()
      terminate = False

    elif (LA.norm(self.state[0:2]) < tol):
      if (self.function_nb == nb_functions - 1):
        terminate = True
        self.function_nb = 0
        self.nb_passes += 1
        print('number of passes:')
        print(self.nb_passes)
      else:
        print(self.function_nb)
        self.function_nb += 1
        self.reset()
        terminate = False
    else:
      terminate = False

    # Set placeholder for info
    info = {}
    
    # Return step information
    return self.state, reward, terminate, info
  
  def render(self):
    pass

  def get_function_nb(self):
    return self.function_nb

  def reset(self):
    ini_x = -5. + 10*np.random.rand()
    ini_y = -5. + 10*np.random.rand()
    # ini_x = 5.
    # ini_y = 5.
    ini = np.array([ini_x, ini_y])
    quadobj = Objective(self.function_nb)
    func_val = quadobj.get_fval(ini)
    jac_eval = quadobj.get_jacval(ini)
    self.state = np.array([ini_x, ini_y, func_val, jac_eval[0], jac_eval[1]])
    self.iterations = 0
    
    quadobj = Objective(self.function_nb)
    Q = quadobj.get_Q()
    eigs, _ = LA.eig(Q)
    max_step = 2./np.max(eigs)
    self.action_space = Box(low=0., high=max_step, shape=(1,), dtype=np.float64)
    return self.state
  