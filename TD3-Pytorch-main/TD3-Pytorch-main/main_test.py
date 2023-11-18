from gradient_descent import gradient_descent
import numpy as np
from numpy import linalg as LA
from visualizer import visualize
import matplotlib.pyplot as plt
from objective import Objective
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import gd_env

function_nb = 0

quadobj = Objective(function_nb)
Q = quadobj.get_Q()
eigs, _ = LA.eig(Q)
dimension = np.size(Q,1)
X = 5.*np.ones((1,dimension))
traj,iter, fe_cache = gradient_descent(X, function_nb)
traj = np.reshape(traj,(int(np.size(traj)/dimension),dimension))


traj_ls,iter_ls, fe_cache_ls = gradient_descent(X,function_nb,True)
traj_ls = np.reshape(traj_ls,(int(np.size(traj_ls)/dimension),dimension))


# filename = "TD3_data.txt"
# file1 = open(filename, "r")
# lines = file1.readlines()
# for i in range(np.size(lines)):
#     if i==0:
#         TD3_data = np.fromstring(lines[i], dtype=float, sep=' ')
#     else:
#         line = np.fromstring(lines[i], dtype=float, sep=' ')
#         TD3_data = np.append(TD3_data,line)
# TD3_data = np.reshape(TD3_data,(int(np.size(TD3_data)/2),2))

# filename2 = "TD3_trajectory.txt"
# file2 = open(filename2, "r")
# lines2 = file2.readlines()
# for i in range(np.size(lines2)):
#     if i==0:
#         TD3_traj = np.fromstring(lines2[i], dtype=float, sep=' ')
#     else:
#         line2 = np.fromstring(lines2[i], dtype=float, sep=' ')
#         TD3_traj = np.append(TD3_traj,line2)
# TD3_traj = np.reshape(TD3_traj,(int(np.size(TD3_traj)/2),2))

# env = make_vec_env("gd_env-v0", n_envs=1, seed=0, mode='test')
env = gym.make("gd_env-v0", mode='test')
env = DummyVecEnv([lambda: env])
env.reset()
model = TD3.load("gd",env=env)

obs = env.reset()
obs_ = obs[0]
done = False
TD3_traj = np.array(obs_[0:dimension])
action_cache = []
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_cache = np.append(action_cache,action)
    obs, rewards, done, info = env.step(action)
    obs_ = obs[0]
    TD3_traj = np.append(TD3_traj,obs_[0:dimension],axis=0)

TD3_traj = np.reshape(TD3_traj[0:-dimension],(int(np.size(TD3_traj[0:-dimension],0)/dimension),dimension))

TD3_data = np.append(np.linspace(1,np.size(TD3_traj,0),np.size(TD3_traj,0)),LA.norm(TD3_traj,axis=1),axis = 0)
TD3_data = np.reshape(TD3_data,(int(np.size(TD3_data)/2),2),order='F')

visualize(traj[:,0:2], traj_ls[:,0:2], TD3_traj[:,0:2], function_nb)
plt.clf()
plt.semilogy(np.linspace(1,np.size(traj,0),np.size(traj,0)),LA.norm(traj,axis=1),label='GD')
plt.semilogy(np.linspace(1,np.size(traj_ls,0),np.size(traj_ls,0)),LA.norm(traj_ls,axis=1),label='LS')
plt.semilogy(TD3_data[0:,0], TD3_data[0:,1],'g--',label='TD3')
plt.xlabel('iterations')
plt.ylabel('||abs error||')
plt.legend(loc="upper right")
plt.grid()
# plt.show()
plt.savefig('convergence.png')

fe_cache_td3 = ((dimension+1)/dimension)*fe_cache

plt.clf()
plt.semilogy(fe_cache,LA.norm(traj,axis=1),label='GD')
plt.semilogy(fe_cache_ls,LA.norm(traj_ls,axis=1),label='LS')
plt.semilogy(np.array([0,fe_cache_td3[-1]]), np.array([TD3_data[0,1], TD3_data[-1,1]]),'g--',label='TD3')
plt.xlabel('function evaluations')
plt.ylabel('||abs error||')
plt.legend(loc="upper right")
plt.grid()
# plt.show()
plt.savefig('function_eval.png')

plt.clf()
plt.plot(TD3_data[0:,0],action_cache)
plt.xlabel('iterations')
plt.ylabel('action')
plt.grid()
plt.savefig('actions.png')

