import numpy as np
import matplotlib.pyplot as plt

from AcuteMesh import Mesh
from FracMFGSolver import MFGSolver

l = 2
w = 0.5*l
h = 0.5*l*np.tan(3/8*np.pi)
d = 0.3*w

p = np.array([
    [-w,h],
    [w,h],
    [h,w],
    [h,-w],
    [w,-h],
    [-w,-h],
    [-h,-w],
    [-h,w],
    [-0.3*h,0.3*h],
    [0.4*h,0],
    [0,-0.4*h],
    [0.3*h + np.sqrt(2)*d,-np.sqrt(2)*d-d],
    [np.sqrt(2)*d,-0.4*h-np.sqrt(2)*d],
    [h,-0.1*w],
    [0.5*(h+w),-0.5*(w+h)]
]).T


t = np.array([
    [0,1,8],
    [1,8,9],
    [1,2,9],
    [2,9,13],
    [9,11,13],
    [11,13,3],
    [3,11,14],
    [4,12,14],
    [11,12,14],
    [5,4,12],
    [5,10,12],
    [5,6,10],
    [6,7,8],
    [8,9,10],
    [6,8,10]
    
])

bndrynode = np.ones(p.shape[1], dtype = np.bool)

e = np.array([
    [0,1],
    [1,2],
    [2,3],
    [2,13],
    [13,3],
    [3,14],
    [14,4],
    [4,5],
    [5,6],
    [6,7],
    [7,8],
    [8,0],
    [9,10],
    [10,12],
    [12,11],
    [11,9],
])

normals = np.array([
    [p[1,edge[1]]-p[1,edge[0]],p[0,edge[0]]-p[0,edge[1]]] for edge in e
])


mesh = Mesh(p,t,e,bndrynode,normals)


mesh.Refine(4)
mesh.ComputeSizes()

#mesh.PlotMesh(highlight_bndrynodes=True, numerate_nodes=True)

eps = 0
nu = 1
kappa = 1


Fm = lambda m : np.tanh(m)
F0 = lambda x : (np.cos(np.pi*x[0]) + np.sin(np.pi*x[1]))*np.exp(x[0]*x[1])
def F1(p):
    # generated with ChatGPT

    single_vector = False
    if p.ndim == 1:
        p = p.reshape(2, 1)
        single_vector = True

    x, y = p[0], p[1]
    
    indicators = np.zeros(p.shape[1], dtype=int)

    # Assign quadrant values
    indicators[(x >= 0) & (y >= 0)] = 0  # Quadrant I
    indicators[(x < 0) & (y > 0)] = 1  # Quadrant II
    indicators[(x <= 0) & (y <= 0)] = 0  # Quadrant III
    indicators[(x > 0) & (y < 0)] = 1  # Quadrant IV

    if single_vector:
        return p*indicators[0]
    return p*indicators

F = [F0,F1]

H = lambda p : np.linalg.norm(p,axis = 0)
def dH(x, axis=-1, eps=1e-10):
    """
    Normalizes a NumPy array of 2D vectors.

    Parameters:
        vectors: np.ndarray of shape (..., 2), where the last dimension is the vector components.
        axis: Axis along which to normalize. Default is -1 (last axis).
        eps: Small epsilon to avoid division by zero.

    Returns:
        np.ndarray of same shape as `vectors` with normalized vectors.
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + eps)

def G0(p):
    # generated with ChatGPT

    single_vector = False
    if p.ndim == 1:
        p = p.reshape(2, 1)
        single_vector = True

    x, y = p[0], p[1]
    
    indicators = np.zeros(p.shape[1], dtype=int)

    # Assign quadrant values
    indicators[(x >= 0) & (y >= 0)] = 1  # Quadrant I
    indicators[(x < 0) & (y > 0)] = 2  # Quadrant II
    indicators[(x <= 0) & (y <= 0)] = 3  # Quadrant III
    indicators[(x > 0) & (y < 0)] = 4  # Quadrant IV

    if single_vector:
        return indicators[0]
    return indicators
G = [G0,None]

s = 0.75

FracMFG = MFGSolver(mesh,eps,nu,kappa)

uh,mh = FracMFG.Solve(H,dH,Fm,F,G,s,mu = 0,C_H = 1)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(mesh.p[0,:], mesh.p[1,:], mh, triangles=mesh.t, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.show()