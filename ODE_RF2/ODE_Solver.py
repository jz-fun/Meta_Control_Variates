
import numpy as np


def ODE_Solver(X):
    '''
    :param X: \omega
    :return: fine approximation
    '''

    a=X[:,0]
    z=X[:,1]
    N=int(np.shape(a)[0])

    #set grid size
    n=32

    #Step2: ODE solvers:
    Pf = np.zeros(N)  # fine approximation
    nf = n
    hf = 1/nf
    l=2


    if l>1:
        #fine
        #A0f
        cf = np.repeat(1,nf)
        A0f = np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf - 1), (nf - 1)))
        A0f[(grid[0] - grid[1] == 1)|(grid[0] - grid[1] == -1)]=1
        A0f=hf**(-2)*A0f

        #A1f
        cf=(np.array(range(1,(nf+1)))-0.5)*hf
        A1f=np.diag(-cf[1:] - cf[:(nf-1)])
        grid = np.indices(((nf-1),(nf-1)))
        A1f[grid[0] - grid[1] == 1]=cf[1:nf-1]
        A1f[grid[0] - grid[1] == -1]=cf[1:nf-1]
        A1f=hf**(-2)*A1f

        cf = np.repeat(1, nf-1)

        for nl in range(0,N):
            U = a[nl]
            Z = z[nl]
            uf=np.linalg.inv(A0f+U*A1f)*(50*Z**2*cf)
            Pf[nl]=np.sum(hf*uf)



    return Pf






