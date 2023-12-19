import numpy as np
import matplotlib.pyplot as plt
eig = 1 + (10-1)*np.random.rand(100,1)
eig[0] = 1
eig[1] = 10
Q = np.diagflat(eig)
Q_all = np.reshape(Q,(1,int(np.size(Q)/1)))
np.savetxt("test_func.txt", Q_all, fmt='%4.15f', delimiter=' ')

q = 1 + (10-1)*np.random.rand(100,1)
q = np.reshape(q,(1,int(np.size(q)/1)))
np.savetxt("test_func_q.txt", q, fmt='%4.15f', delimiter=' ')


# x1 = np.linspace(-10.0, 10.0, 100)
# x2 = np.linspace(-10.0, 10.0, 100)
# X1, X2 = np.meshgrid(x1, x2)
# y = np.concatenate((X1,X2),axis=0)
# Q = np.array([[1. ,0.],[0., 5.]])
# X = y
# var = 2
# X_rows = np.size(X,0)
# var_rows = int(X_rows/var)
# temp = np.zeros((100,var*100))
# for i in range(0,var):
#     for j in range(0,var):
#         temp[:,j*100:(j+1)*100] += X[var_rows*j:var_rows*(j+1),:]*Q[j,i]
# fval = 0
# for i in range(0,var):
#     fval += temp[:,i*100:(i+1)*100]*X[var_rows*i:var_rows*(i+1),:]
# fval = 0.5*fval
# Y = fval


# cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
# plt.clabel(cp, inline=1, fontsize=10)
# cp = plt.contourf(X1, X2, Y, )
# plt.savefig("contour.png")
