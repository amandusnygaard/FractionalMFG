import numpy as np
from scipy.sparse import coo_matrix

from AcuteMesh import Mesh

def ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True):

    if inverse:
        return np.linalg.inv(np.vstack((x1-x0,x2-x0)).T)
    return np.vstack((x1-x0,x2-x0)).T

class GridFunction:

    def __init__(self, mesh, coeff):
        '''
        mesh : Mesh object that gridfunction is defined on
        uh : value of gridfunction on nodes. Also coefficients for basis expansion with tent functions
        '''

        self.mesh = mesh

        self.coeff = coeff

        return None
    

phi = [lambda x : 1-x[0]-x[1], lambda x: x[0], lambda x : x[1]]
phi_grad = [np.array([-1,-1]),np.array([1,0]),np.array([0,1])]

###################################################
# Quadrature
w = 0.5*np.array([0.225,0.132394152788506,0.132394152788506,0.132394152788506,
                    0.125939180544827,0.125939180544827,0.125939180544827])

p = np.vstack((np.array([0.333333333333333,0.470142064105115,0.470142064105115,
                        0.059715871789770,0.101286507323456,0.101286507323456,
                        0.797426985353087]),
               np.array([0.333333333333333,0.059715871789770,0.470142064105115,
                        0.470142064105115,0.797426985353087,0.101286507323456,
                        0.101286507323456])))

n_q = len(w)

def Quad(f, A, j):

    value = 0

    for i in range(n_q):

        value += w[i]*f(A(p[:,i]))*phi[j](p[:,i])

    return value

def QuadGrad(df, A, j = None):

    value = 0

    if j is None:

        for i in range(n_q):

            value += w[i]*df(A(p[:,i]))

        return value

    for i in range(n_q):

        value += w[i]*df(A(p[:,i]))*phi[j](p[:,i])

    return value


def LocalIntegrateL2(u,uh,mesh, i):

    T = mesh.t[i]

    x0 = mesh.p[:,T[0]]
    x1 = mesh.p[:,T[1]]
    x2 = mesh.p[:,T[2]]

    c = 2*mesh.area[i]

    K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse=False)

    I = 0

    for i in range(n_q):
        y = p[:,i]
        z = K@y+ x0
        I += w[i]*(u(z) - uh[T[0]]*phi[0](y)-uh[T[1]]*phi[1](y)-uh[T[2]]*phi[2](y))**2

    return c*I

def LocalIntegrateH1(Du,uh,mesh,i):

    T = mesh.t[i]

    x0 = mesh.p[:,T[0]]
    x1 = mesh.p[:,T[1]]
    x2 = mesh.p[:,T[2]]


    c = 2*mesh.area[i]

    K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse=False)
    Kinv = np.linalg.inv(K)

    I = 0

    for i in range(n_q):
        y = p[:,i]
        z = K@y+ x0

        De = Du(z) - Kinv.T@(uh[T[0]]*phi_grad[0]+uh[T[1]]*phi_grad[1]+uh[T[2]]*phi_grad[2])

        I += w[i]*np.inner(De,De)

    return c*I

###################################################

def IntegrateL2(u,uh,mesh, root = True):

    I = 0

    for i in range(mesh.N_T):

        I += LocalIntegrateL2(u,uh,mesh,i)

    if root:
        return np.sqrt(I)
    
    return I

def IntegrateH1(Du,uh,mesh, root = True):

    I = 0

    for i in range(mesh.N_T):

        I += LocalIntegrateH1(Du,uh,mesh,i)

    if root:
        return np.sqrt(I)
    
    return I

from scipy.sparse import coo_matrix

def compute_phi_on_refined_mesh_sparse(p, t, p_ref):
    N_ref = p_ref.shape[1]
    N_p = p.shape[1]

    rows = []
    cols = []
    data = []

    assigned = np.zeros(N_ref, dtype=bool)

    for tri in t:
        i, j, k = tri
        p0, p1, p2 = p[:, i], p[:, j], p[:, k]

        v0 = p1 - p0
        v1 = p2 - p0
        mat = np.column_stack((v0, v1))  # (2, 2)
        try:
            inv_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            continue  # skip degenerate triangle

        rel = p_ref.T - p0  # (N_ref, 2)
        bary = rel @ inv_mat.T  # shape: (N_ref, 2)
        lambda2 = bary[:, 0]
        lambda3 = bary[:, 1]
        lambda1 = 1.0 - lambda2 - lambda3

        # Unassigned and inside triangle (with tolerance)
        inside = (lambda1 >= -1e-10) & (lambda2 >= -1e-10) & (lambda3 >= -1e-10) & (~assigned)

        idx_inside = np.where(inside)[0]
        if len(idx_inside) == 0:
            continue

        rows.extend(idx_inside.tolist() * 3)
        cols.extend(([i] * len(idx_inside)) + ([j] * len(idx_inside)) + ([k] * len(idx_inside)))
        data.extend(lambda1[inside].tolist() + lambda2[inside].tolist() + lambda3[inside].tolist())

        # Mark as assigned
        assigned[idx_inside] = True

    phi_sparse = coo_matrix((data, (rows, cols)), shape=(N_ref, N_p))
    return phi_sparse.tocsr()
