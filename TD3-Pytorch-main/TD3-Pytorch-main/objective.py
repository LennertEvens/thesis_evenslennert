import numpy as np
from numpy import linalg as LA

class Objective:
    def __init__(self, function_nb, mode='train'):
        
        # if mode == 'train':
        #     filename = "functions.txt"

        # else:
        #     filename = "eval_set.txt"
        # file1 = open(filename, "r")
        # lines = file1.readlines()
        # Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
        # Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))
        if mode == 'test':
            filename = "eval_set.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
            Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))
        elif mode == 'eval':
            filename = "eval_set.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
            Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))

            eig = 1 + 2*np.random.rand()
            Q = np.array([[1., 0.],[0., eig]])
        else:
            eig = 1 + 2*np.random.rand()
            Q = np.array([[1., 0.],[0., eig]])

        self.Q = Q

    def get_Q(self):
        return self.Q
    
    def get_fval(self, X, visualize = False):
        if visualize:
            var = 2
            X_rows = np.size(X,0)
            var_rows = int(X_rows/var)
            temp = np.zeros((100,var*100))
            for i in range(0,var):
                for j in range(0,var):
                    temp[:,j*100:(j+1)*100] += X[var_rows*j:var_rows*(j+1),:]*self.Q[j,i]
            fval = 0
            for i in range(0,var):
                fval += temp[:,i*100:(i+1)*100]*X[var_rows*i:var_rows*(i+1),:]
            fval = 0.5*fval

        else:
            f = np.matmul(X,self.Q)
            fval = 0.5*np.matmul(f,np.transpose(X))

        return fval
    
    def get_jacval(self, X):
        jacval = np.matmul(X,self.Q)
        return jacval
    
    def get_max_step(self):
        eigs, _ = LA.eig(self.Q)
        eigs = np.real(eigs)
        max_step = float(2./np.max(eigs))
        return max_step





        
    