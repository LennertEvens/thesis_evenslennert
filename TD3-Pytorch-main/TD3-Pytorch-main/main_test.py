from gradient_descent import gradient_descent
import numpy as np
from numpy import linalg as LA
from visualizer import visualize
import matplotlib.pyplot as plt
from objective import Objective

function_nb = 0

quadobj = Objective(function_nb)
eigs, _ = LA.eig(quadobj.get_Q())
X = np.array([5.0 , 5.0])
traj,iter, fe_cache = gradient_descent(X, function_nb)
traj = np.reshape(traj,(int(np.size(traj)/2),2))


traj_ls,iter_ls, fe_cache_ls = gradient_descent(X,function_nb,True)
traj_ls = np.reshape(traj_ls,(int(np.size(traj_ls)/2),2))


filename = "TD3_data.txt"
file1 = open(filename, "r")
lines = file1.readlines()
for i in range(np.size(lines)):
    if i==0:
        TD3_data = np.fromstring(lines[i], dtype=float, sep=' ')
    else:
        line = np.fromstring(lines[i], dtype=float, sep=' ')
        TD3_data = np.append(TD3_data,line)
TD3_data = np.reshape(TD3_data,(int(np.size(TD3_data)/2),2))

filename2 = "TD3_trajectory.txt"
file2 = open(filename2, "r")
lines2 = file2.readlines()
for i in range(np.size(lines2)):
    if i==0:
        TD3_traj = np.fromstring(lines2[i], dtype=float, sep=' ')
    else:
        line2 = np.fromstring(lines2[i], dtype=float, sep=' ')
        TD3_traj = np.append(TD3_traj,line2)
TD3_traj = np.reshape(TD3_traj,(int(np.size(TD3_traj)/2),2))

visualize(traj, traj_ls, TD3_traj, function_nb)
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

fe_cache_td3 = (3/2)*fe_cache

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

steps = np.arange(0,65000,5000)
rewards = np.array([-6111139.176610323, -1396808.5075511464, -311669.2697258588, 195995.99267882403, 495956.03111633117, 682934.286295848, 827354.6298671749, 892762.981866117, 818918.9397542115, 615557.7300825403, 281947.6638586336,-286539.8331692803,-1619059.846731976])
plt.clf()
plt.plot(steps,rewards)
plt.grid()
plt.xlabel('Training steps')
plt.ylabel('Reward on val set')
plt.savefig('rewards.png')
