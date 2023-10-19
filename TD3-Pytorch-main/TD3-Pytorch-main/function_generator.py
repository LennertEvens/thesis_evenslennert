import numpy as np

filename = "functions.txt"
nb_functions = 100

def generate_Q():
    alpha = -np.pi/2 + np.pi*np.random.rand()
    beta = -np.pi/2 + np.pi*np.random.rand()
    alpha = 0.
    beta = 0.
    U1 = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    U2 = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])
    Q = np.array([[1, 0],[0, 1+(10-1)*np.random.rand()]])
    Q = np.matmul(U1,Q)
    Q = np.matmul(Q,U2)
    return Q


for i in range(nb_functions):
    Q = generate_Q()
    
    if i==0:
        Q_all = np.ndarray.flatten(Q)
    else:
        Q_all = np.append(Q_all,Q)
Q_all = np.reshape(Q_all,(nb_functions,int(np.size(Q_all)/nb_functions)))
np.savetxt(filename, Q_all, fmt='%4.15f', delimiter=' ') 



