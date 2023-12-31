import numpy as np
mode = 1
nb_train = 500
nb_eval = 100
if mode == 1:
    filename = "functions.txt"
    nb_functions = nb_train
else:
    filename = "eval_set.txt"
    nb_functions = nb_eval

def generate_Q(alpha, a):
    # alpha = -np.pi/2 + np.pi*np.random.rand()
    # beta = -np.pi/2 + np.pi*np.random.rand()
    # alpha = 0.
    # beta = 0.
    U1 = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    # U2 = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
    # Q = np.array([[1, 0],[0, 1+(10-1)*np.random.rand()]])
    Q = np.array([[1, 0],[0, a]])
    Q = np.matmul(U1,Q)
    Q = np.matmul(Q,np.transpose(U1))
    return Q

alpha_array = np.linspace(0,np.pi,nb_functions)
a_array = np.linspace(1,10,nb_functions)
for i in range(nb_functions):
    alpha = alpha_array[i]
    a = a_array[i]
    Q = generate_Q(alpha, a)
    
    if i==0:
        Q_all = np.ndarray.flatten(Q)
    else:
        Q_all = np.append(Q_all,Q)
Q_all = np.reshape(Q_all,(nb_functions,int(np.size(Q_all)/nb_functions)))
np.savetxt(filename, Q_all, fmt='%4.15f', delimiter=' ') 



