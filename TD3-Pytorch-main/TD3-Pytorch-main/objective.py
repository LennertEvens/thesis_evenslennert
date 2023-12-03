import numpy as np
from numpy import linalg as LA

class Objective:
    def __init__(self, function_nb=0, mode='train'):
        
        # if mode == 'train':
        #     filename = "functions.txt"

        # else:
        #     filename = "eval_set.txt"
        # file1 = open(filename, "r")
        # lines = file1.readlines()
        # Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
        # Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))
        if mode == 'test':
            # Q = np.array([[50., 0.],[0., 80.]])
            # dia = np.array([1., 8.0, 4.0, 7.5, 5.2, 10., 4.8, 3.2, 8.3, 6.8])
            # Q = np.diagflat(dia)
            # eig = 1 + (100-1)*np.random.rand(10,1)
            # Q = np.diagflat(eig)
            filename = "test_func.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            Q = np.fromstring(lines[0], dtype=float, sep=' ')
            Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))

            filename = "test_func_q.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            q = np.fromstring(lines[0], dtype=float, sep=' ')
            q = np.reshape(q,(np.size(q),1))

            # Q = np.matmul(U,Q)
            # Q = np.matmul(Q,np.transpose(U))

        elif mode == 'eval':
            # filename = "eval_set.txt"
            # file1 = open(filename, "r")
            # lines = file1.readlines()
            # Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
            # Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))
            eig = 1 + (10-1)*np.random.rand(10,1)
            Q = np.diagflat(eig)
            filename = "test_func.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            Q = np.fromstring(lines[0], dtype=float, sep=' ')
            Q = np.reshape(Q,(int(np.sqrt(np.size(Q))),int(np.sqrt(np.size(Q)))))

            filename = "test_func_q.txt"
            file1 = open(filename, "r")
            lines = file1.readlines()
            q = np.fromstring(lines[0], dtype=float, sep=' ')
            q = np.reshape(q,(np.size(q),1))
            # dia = np.array([1., 8.0, 4.0, 7.5, 5.2, 10., 4.8, 3.2, 8.3, 6.8])
            # Q = np.diag(dia)
            # Q = np.array([[1., 0.],[0., 10.]])
            # filename = "rotation.txt"
            # file1 = open(filename, "r")
            # lines = file1.readlines()
            # U = np.fromstring(lines[0], dtype=float, sep=' ')
            # U = np.reshape(U,(int(np.sqrt(np.size(U))),int(np.sqrt(np.size(U)))))
            # Q = np.matmul(U,Q)
            # Q = np.matmul(Q,np.transpose(U))

        else:
            # eig = 1 + (10-1)*np.random.rand()
            # Q = np.array([[1., 0.],[0., eig]])
            eig = 1 + (10-1)*np.random.rand(100,1)
            Q = np.diagflat(eig)
            q = 1 + (10-1)*np.random.rand(100,1)
            # dia = np.array([1., 8.0, 4.0, 7.5, 5.2, 10., 4.8, 3.2, 8.3, 6.8])
            # Q = np.diag(dia)
            # Q = np.array([[1., 0.],[0., 10.]])
            # filename = "rotation.txt"
            # file1 = open(filename, "r")
            # lines = file1.readlines()
            # U = np.fromstring(lines[0], dtype=float, sep=' ')
            # U = np.reshape(U,(int(np.sqrt(np.size(U))),int(np.sqrt(np.size(U)))))
            # Q = np.matmul(U,Q)
            # Q = np.matmul(Q,np.transpose(U))
        self.Q = Q
        self.q = q
        self.q = np.zeros(100)

    def get_Q(self):
        return self.Q
    
    def get_fval(self, X, visualize = False):
        if visualize:
            var = 2
            nb = 1000
            X_rows = np.size(X,0)
            var_rows = int(X_rows/var)
            temp = np.zeros((nb,var*nb))
            for i in range(0,var):
                for j in range(0,var):
                    temp[:,j*nb:(j+1)*nb] += X[var_rows*j:var_rows*(j+1),:]*self.Q[j,i]
            fval = 0
            for i in range(0,var):
                fval += temp[:,i*nb:(i+1)*nb]*X[var_rows*i:var_rows*(i+1),:] + X[var_rows*i:var_rows*(i+1),:]*self.q[i]
            fval = 0.5*fval

        else:
            f = np.matmul(X,self.Q)
            fval = 0.5*np.matmul(f,np.transpose(X)) + np.matmul(X,self.q)

        return fval
    
    def get_jacval(self, X):
        jacval = np.matmul(X,self.Q) + self.q
        return jacval
    
    def get_max_step(self):
        eigs, _ = LA.eig(self.Q)
        eigs = np.real(eigs)
        max_step = float(2./np.max(eigs))
        return max_step





        
    