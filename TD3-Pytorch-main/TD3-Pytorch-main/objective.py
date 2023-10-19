import numpy as np

class Objective:
    def __init__(self, function_nb):
        
        filename = "functions.txt"
        file1 = open(filename, "r")
        lines = file1.readlines()
        Q = np.fromstring(lines[function_nb], dtype=float, sep=' ')
        Q = np.reshape(Q,(2,2))

        self.Q = Q

    def get_Q(self):
        return self.Q
    
    def get_fval(self, X, visualize = False):
        if visualize:
            var = np.size(self.Q,0)
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





        
    