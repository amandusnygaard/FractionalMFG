import numpy as np
import matplotlib.pyplot as plt

from AcuteMesh import Mesh
from FracMFGHelper import *

from FracMFGSolver import * 


# N_p = 9
# elements = np.array([[0,1,3],[1,3,4],[1,2,4],[2,4,5],[3,4,6],[4,6,7],[4,5,7],[5,7,8]])
# points = np.array([[0,0],[0,0.5],[0,1],[0.5,0],[0.5,0.5],[0.5,1],[1,0],[1,0.5],[1,1]]).T
# bndrynodes = np.array([1,1,1,1,0,1,1,1,1], dtype = np.bool)

points = np.array([(0,0),(1,0),(0,1),(1,1),
                   (0,0.540),(0.355,0.355),(0.540,0),(0.693,0.307),
                   (1,0.460),(0.640,0.640),(0.460,1),(0.307,0.693)], dtype = np.float64).T

elements = np.array([(0,5,4),(0,6,5),(5,6,7),(6,1,7),
                     (1,8,7),(3,9,8),(9,7,8),(9,5,7),
                     (3,10,9),(10,11,9),(2,11,10),(2,4,11),
                     (4,5,11),(5,9,11)], dtype = np.int32)

edges = np.array([(0,4),(4,2),(2,10),(10,3),(3,8),(8,1),(1,6),(6,0)], dtype = np.int32)

bndrynode = np.array([1,1,1,1,1,0,1,0,1,0,1,0], dtype = np.bool)

normals = np.array([(1,0),(1,0),(0,-1),(0,-1),(-1,0),(-1,0),(0,1),(0,1)])

nu = 1.0
kappa = 1.0
eps = 1.0

mesh = Mesh(points, elements,edges, bndrynode,normals)
# mesh.ComputeTheta()
# theta = mesh.theta
# mesh.Refine(3)
mesh.ComputeSizes()
print(np.max(mesh.h))
# mesh.FindPatches()

# testMFG = MFGSolver(mesh,eps,nu,kappa)

# u = sin(pi x)sin(pi y). dHU = (grad u)/sqrt(1+|grad u|^2)
u = lambda x : np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
Du = lambda x : np.array([np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]),
                          np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])
m = lambda x : x[0]*(1-x[0]**2)*x[1]*(1-x[1]**2)
Dm = lambda x : np.array([(1-3*x[0]**2)*x[1]*(1-x[1]**2),
                          (1-3*x[1]**2)*x[0]*(1-x[0]**2)])

L2_u = 0.5
L2_m = np.sqrt(64/11025)
H1_u = np.sqrt(np.pi**2/2 + L2_u**2)
H1_m = np.sqrt(64/525 + L2_m**2)

dHU = lambda x: (np.array([np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]),
                          np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])])*
                          1/np.sqrt(1+np.pi**2*(np.cos(np.pi*x[0])**2*np.sin(np.pi*x[1])**2 + 
                                                np.sin(np.pi*x[0])**2*np.cos(np.pi*x[1])**2)))

dH = lambda p : p/np.sqrt(1+np.inner(p,p))
H = lambda p : np.sqrt(1+np.inner(p,p))-1

G0 = lambda x : 6*eps*(x[1]*x[0]*(1-x[0]**2)+x[0]*x[1]*(1-x[1]**2)) + kappa*m(x)
G1 = lambda x : dHU(x)*m(x)

Fm = lambda m : np.tanh(m)
F0 = lambda x : (2*np.pi**2*eps+kappa)*u(x) + H(Du(x)) - Fm(m(x))
F1 = None

F = [F0,F1]
G = [G0,G1]

# Need to run FracPreCalc.py first if you want to include the fractional
# Laplace term in the right hand side with the same mesh
data = np.load('FracLapl_075_5ref.npz')

err_u_L2,err_u_H1,err_m_L2,err_m_H1, h_list,stabilization = ConvergenceTest(mesh,eps,nu,kappa,H,dH,Fm,F,G,
                                                              sol = [u,m],diff_sol = [Du,Dm], sol_norms = [L2_u,H1_u,L2_m,H1_m],
                                                              As = data['fraclapl'], s = data['s'], p_ref = data['p_ref'],
                                                              num_refs=4, start_ref=0,mu = 0,)

print(err_u_L2)
print(err_m_L2)
print(err_u_H1)
print(err_m_H1)
print(h_list)
print(stabilization)