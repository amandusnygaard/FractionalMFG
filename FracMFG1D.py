import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma,rgamma

from AcuteMesh import Mesh1D
from FracMFG1DSolver import MFGSolver1D, ConvergenceTest
from FracMFG1DHelper import IntegrateL2, IntegrateH1, StiffnessFractionalLaplaceUniform

# s = 0.8
# cns = 2**(-2*s)*np.sqrt(np.pi)*rgamma(0.5*(1+2*s))*rgamma(1+s)
# u = lambda x : cns*(L**2-x**2)**s
# f = lambda x : 1 + u(x)
k = 1
N = 10*2**k
L = 1
x = np.linspace(-L,L,N+1)

kappa = 1
eps = 0#1*2**(-k)
nu = 1

s = 0.75

mesh_ref = Mesh1D(x)
mesh_ref.Refine(9)

As = StiffnessFractionalLaplaceUniform(mesh_ref.p,mesh_ref.h,s)

mesh = Mesh1D(x)
# mesh.Refine(5)

H = lambda p : np.sqrt(1+p*p)-1
dH = lambda p : p/np.sqrt(1+p*p)
# m = lambda x : (L**2-x**2)*np.exp(-x)
# dm = lambda x : np.exp(-x)*(x**2-2*x-L**2)
m = lambda x : L**2-x**2
dm = lambda x : -2*x
u = lambda x : np.sin(np.pi*x/L)
du = lambda x : np.pi/L*np.cos(np.pi*x/L)
dHU = lambda x : dH(du(x))

L2_u = 4*np.sqrt(L**5/15)
H1_u = np.sqrt(8*L**3/3 + L2_u**2)
L2_m = np.sqrt(L)
H1_m = np.sqrt(np.pi**2/L + L)


Fm = lambda m : np.tanh(m)
FmM = lambda x : Fm(m(x))
F = lambda x : (kappa+eps*np.pi**2/L**2)*u(x) + H(du(x)) - FmM(x)

# G0 = lambda x : eps*np.exp(-x)*(x**2-4*x-L**2+2) + kappa*m(x)
G0 = lambda x : 2*eps + kappa*m(x)
G1 = lambda x : dH(du(x))*m(x)

G = [G0,G1]


test = MFGSolver1D(mesh,nu,kappa,eps)

uh,mh = test.Solve(H,dH,Fm,F,G, s = s,As = As,p_ref = mesh_ref.p, func_ex = [u,m])

plt.plot(mesh.p,uh)
plt.plot(mesh.p,u(mesh.p), c = 'k')
plt.show()

plt.plot(mesh.p,mh)
plt.plot(mesh.p,m(mesh.p), c = 'k')
plt.show()

plt.plot(mesh.p,uh-u(mesh.p))
plt.show()

plt.plot(mesh.p,mh-m(mesh.p))
plt.show()

print(IntegrateL2(u,uh,mesh))
print(IntegrateH1(du,uh,mesh))
print(IntegrateL2(m,mh,mesh))
print(IntegrateH1(dm,mh,mesh))

err_u_L2,err_u_H1,err_m_L2,err_m_H1, h_list,stabilization = ConvergenceTest(mesh,eps,nu,kappa,H,dH,Fm,F,G,
                                                              sol = [u,m],diff_sol = [du,dm], sol_norms = [L2_u,H1_u,L2_m,H1_m],
                                                              As = As, s = s, p_ref = mesh_ref.p,
                                                              num_refs=6, start_ref=0,
                                                              filename = None)

print(err_u_L2)
print(err_m_L2)
print(err_u_H1)
print(err_m_H1)
print(h_list)
print(stabilization)

plt.figure()
plt.loglog(h_list,err_u_L2, label = 'uhs L2', marker = 'o')
plt.loglog(h_list,err_u_H1, label = 'uhs H1', marker = 'o')
plt.loglog(h_list,err_m_L2, label = 'mhs L2', marker = 'o')
plt.loglog(h_list,err_m_H1, label = 'uhs H1', marker = 'o')
plt.grid(True)
plt.legend()
plt.show()
