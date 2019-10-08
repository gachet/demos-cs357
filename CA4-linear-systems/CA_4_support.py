import numpy as np
import scipy.sparse as sparse

def gll(p):
    # creates Gauss-Lobatto-Legendre integration points on interval [-1,1] and corresponding weights
    n = p+1

    z = np.zeros(n)
    w = np.zeros(n)
    z[0] = -1
    z[-1] = 1
    M = np.zeros((p-1,p-1))
    for i in range(p-2):
        k = i+1
        M[i,k] = 0.5*np.sqrt((k*(k+2))/((k+0.5)*(k+1.5)))
        M[k,i] = M[i,k]

    z[1:-1] = np.sort(np.linalg.eig(M)[0])

    w[0] = 2/(p*n)
    w[-1] = w[0]
    for i in range(1,n-1):
        x = z[i]
        z0 = 1
        z1 = x
        for j in range(1,p):
            z2 = x*z1*(2*j+1)/(j+1) - z0*j/(j+1)
            z0 = z1
            z1 = z2
        w[i] = 2/(p*n*z2*z2)
        
    return z,w

def fd_weights(zz,z):
    # coefficients for spatial derivative
    n1 = z.shape[0]
    n = n1 - 1
    c1 = 1.
    c4 = z[0] - zz
    c = np.zeros((n1,2))
    c[0,0] = 1.

    for i in range(1,n+1):
        c2 = 1.
        c5 = c4
        c4 = z[i] - zz
        for j in range(i):
            c3 = z[i] - z[j]
            c2 = c2*c3
            c[i,1] = c1*(c[i-1,0]  - c5*c[i-1,1])/c2
            c[i,0] = -c1*c5*c[i-1,0]/c2
            c[j,1] = (c4*c[j,1] - c[j,0])/c3
            c[j,0] = c4*c[j,0]/c3
        c1 = c2
    return c

def dhat(z):
    # matrix for computing first derivative of function
    n1 = z.shape[0]
    w = np.zeros((n1,2))
    Dh = np.zeros((n1,n1))
    for i in range(n1):
        w = fd_weights(z[i],z)
        Dh[:,i] = w[:,1]
    return Dh.T

def semhat(N):
    # matrices for spectral element method
    z,w = gll(N)
    Bh = np.diag(w)
    Dh = dhat(z)
    Ah = Dh.T@Bh@Dh
    Ch = Bh@Dh
    return Ah, Bh, Ch, z, w

def SEM_system_1(N):
    Ah,Bh,Ch,x,_ = semhat(N)
    A = Ah + Ch
    A = A[1:-1,1:-1]
    return A,x

def SEM_rhs_1(f,x):
    f_ = f(x)
    N = x.shape[0] - 1
    _,w = gll(N)
    b = w*f_
    b = b[1:-1]
    return b

def SEM_system_2(N,dt):
    Ah,Bh,Ch,x,_ = semhat(N)
    A = Bh + dt*0.1*Ah + dt*Ch
    A = A[1:,1:]
    return A, x

def SEM_rhs_2(u_old):
    N = u_old.shape[0] - 1
    _,w = gll(N)
    b = w*u_old
    return b[1:]

def SEM_system_3(N,dt):
    Ah,Bh,Ch,x,_ = semhat(N)
    A = Bh + dt*0.1*Ah
    A = A[1:-1,1:-1]
    return A, x

def SEM_rhs_3(u_old):
    N = u_old.shape[0] - 1
    _,w = gll(N)
    b = w*u_old
    return b[1:-1]

def FDM_system(N):
    M = N-1
    x = np.linspace(0,1,N)
    y = np.linspace(0,1,N)
    dx = 1/M
    
    X,Y = np.meshgrid(x,y)
    d1 = 4*M*M * np.ones((M-1)**2) + 1
    d2 = M*M*np.tile(np.append(-np.ones(M-2),0),M-1)[:-1]
    d3 = M*M*np.tile(np.append(0,-np.ones(M-2)),M-1)[1:]
    d4 = -M*M*np.ones((M-1)*(M-2))

    A = sparse.diags([d1,d2,d3,d4,d4], [0, 1, -1, M-1, 1 -M])
    return A, X, Y

def FDM_rhs(f,X,Y):
    return f(X,Y)[1:-1,1:-1].ravel()

def solution_to_2D(u):
    N = int(np.sqrt(u.shape[0])) + 2
    U = np.zeros((N,N))
    U[1:-1,1:-1] = u.reshape(N-2,N-2)
    return U