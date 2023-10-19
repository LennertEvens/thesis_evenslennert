import numpy as np
import matplotlib.pyplot as plt
from objective import Objective

def visualize(data, data_ls, data_td3, function_nb):
    x1 = np.linspace(-10.0, 10.0, 100)
    x2 = np.linspace(-10.0, 10.0, 100)
    X1, X2 = np.meshgrid(x1, x2)
    y = np.concatenate((X1,X2),axis=0)
    quadobj = Objective(function_nb)
    Y = quadobj.get_fval(y, True)

    plt.subplot(1,3,1)
    cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(cp, inline=1, fontsize=10)
    cp = plt.contourf(X1, X2, Y, )
    plt.plot(data[:,0],data[:,1],'r-')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Fixed stepsize")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.subplot(1,3,2)
    cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(cp, inline=1, fontsize=10)
    cp = plt.contourf(X1, X2, Y, )
    plt.plot(data_ls[:,0],data_ls[:,1],'r-')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title("Linesearch")
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')  

    plt.subplot(1,3,3)
    cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(cp, inline=1, fontsize=10)
    cp = plt.contourf(X1, X2, Y, )
    plt.plot(data_td3[:,0],data_td3[:,1],'r-')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('TD3')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    

