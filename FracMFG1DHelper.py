import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gamma, rgamma
from scipy.linalg import toeplitz

p, w = leggauss(7)

p = 0.5+0.5*p
w = 0.5*w

# basis functions on interval [0,1] for use in gaussian quadrature
phi = [lambda x : 1-x, lambda x : x]
phi_grad = [-1, 1]


def fquad(f,j,x0,x1):

    return np.inner(np.multiply(f(x0+(x1-x0)*p),phi[j](p)),w)

def quad(f,x0,x1):

    return np.inner(f(x0+(x1-x0)*p),w)

def Fmquad(Fm,Mloc,j):

    return np.inner(np.multiply(Fm(Mloc[0]*phi[0](p)+ Mloc[1]*phi[1](p)),phi[j](p)),w)

def IntegrateL2(u,uh,mesh, root = True):

    I = 0

    N_T = mesh.N_T

    for i in range(N_T):

        x0 = mesh[i]
        x1 = mesh[i+1]

        I += mesh.h[i]*np.inner((u(x0 + (x1-x0)*p)-(uh[i]*phi[0](p)+uh[i+1]*phi[1](p)))**2,w)

    if root:
        return np.sqrt(I)
    
    return I

def IntegrateH1(du,uh,mesh, root = True):
    
    I = 0

    N_T = mesh.N_T

    for i in range(N_T):
        
        x0 = mesh[i]
        x1 = mesh[i+1]

        I += mesh.h[i]*np.inner((du(x0 + (x1-x0)*p)-(uh[i+1]-uh[i])/(x1-x0))**2,w)

    if root:
        return np.sqrt(I)
    
    return I

def IntegrateH12(uh,mesh, root = True):
    I = 0

    N_T = mesh.N_T

    for i in range(N_T):
        
        x0 = mesh[i]
        x1 = mesh[i+1]

        I += mesh.h[i]*((uh[i+1]-uh[i])/(x1-x0))**2

    if root:
        return np.sqrt(I)
    
    return I


def StiffnessFractionalLaplaceUniform(x,h,s):
    '''
    Creates stiffness matrix for the fractional laplace on uniform mesh of order s.
    The stiffness matrix assumes zero boundary conditions
    '''

    N_p = len(x)

    c = np.array([1,-4,6,-4,1])
    p = np.arange(0,N_p-2)

    pi = np.abs(np.vstack((p+2,p+1,p,p-1,p-2)))**(3-2*s)
    t = c@pi*rgamma(4-2*s)*h[0]**(1-2*s)*0.5/np.cos(s*np.pi)

    return toeplitz(t)



from scipy.sparse import coo_matrix

def compute_phi_1d_sparse(p, p_ref):
    """
    Construct sparse interpolation matrix for 1D linear basis functions.

    Parameters:
    - p: (N_p,) or (1, N_p) array of original sorted node positions.
    - p_ref: (N_ref,) or (1, N_ref) array of refined node positions.

    Returns:
    - phi_sparse: (N_ref, N_p) scipy.sparse matrix such that
                  phi_sparse[j, i] = phi_i(p_ref[j])
    """
    p = np.ravel(p)
    p_ref = np.ravel(p_ref)

    N_p = p.shape[0]
    N_ref = p_ref.shape[0]

    # Find which interval each refined point belongs to
    # interval_indices[i] is the index such that p[i] <= p_ref < p[i+1]
    interval_indices = np.searchsorted(p, p_ref, side='right') - 1

    # Clamp to valid range [0, N_p - 2]
    interval_indices = np.clip(interval_indices, 0, N_p - 2)

    i0 = interval_indices
    i1 = i0 + 1

    x0 = p[i0]
    x1 = p[i1]
    h = x1 - x0
    h[h == 0] = 1e-14  # avoid division by zero (shouldn't occur in valid mesh)

    lambda0 = (x1 - p_ref) / h
    lambda1 = (p_ref - x0) / h

    # Build sparse matrix
    rows = np.concatenate([np.arange(N_ref), np.arange(N_ref)])
    cols = np.concatenate([i0, i1])
    data = np.concatenate([lambda0, lambda1])

    phi_sparse = coo_matrix((data, (rows, cols)), shape=(N_ref, N_p))
    return phi_sparse.tocsr()
