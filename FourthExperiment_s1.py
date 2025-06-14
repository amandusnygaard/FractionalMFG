import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma,rgamma

from AcuteMesh import Mesh1D
from FracMFG1DSolver import MFGSolver1D, ConvergenceTest
from FracMFG1DHelper import IntegrateL2, IntegrateH12, StiffnessFractionalLaplaceUniform

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

k = 1
N = 10*2**k
L = 1
x = np.linspace(-L,L,N+1)

mesh_ref = Mesh1D(x)
mesh_ref.Refine(7)

mesh = Mesh1D(x)
mesh.Refine(2)

kappa = 1
nu = 1

FracMFG = MFGSolver1D(mesh,nu,kappa,0)
FracMFG2 = MFGSolver1D(mesh,0,kappa,nu)

n = 8
s = 1 - np.exp(-np.linspace(2, 20, n))
print(s)

H = lambda p : np.sqrt(1+p*p)-1
dH = lambda p : p/np.sqrt(1+p*p)

Fm = lambda m : np.tanh(m)

F = lambda x : np.exp(-x**2)

G0 = lambda x : 1+x
G = [G0,None]

func_ex = [lambda x : np.zeros_like(x), lambda x : np.zeros_like(x)]

err_u_L2 = np.zeros(n)
err_u_H1 = np.zeros(n)
err_m_L2 = np.zeros(n)
err_m_H1 = np.zeros(n)
err_u_max = np.zeros(n)
err_m_max = np.zeros(n)
stabilization = np.zeros(n)

uh,mh = FracMFG2.Solve(H,dH,Fm,F,G,s = None,mu = 0,C_H = 1)

# ss = 0.99999999999
# As_ref = StiffnessFractionalLaplaceUniform(mesh_ref.p,mesh_ref.h,ss)
# uhs,mhs  = FracMFG.Solve(H,dH,Fm,F2,G2,s = ss,mu = 0,C_H = 1,
#                             nonlocaloperator = 'FracLaplUniform', 
#                             As = As_ref,p_ref = mesh_ref.p, func_ex = func_ex)

# plt.plot(mesh.p,uh-uhs)
# plt.show()
# print(np.max(np.abs(uh-uhs)))

# plt.plot(mesh.p,mh-mhs)
# plt.show()
# print(np.max(np.abs(mh-mhs)))

for i in range(n):

    As_ref = StiffnessFractionalLaplaceUniform(mesh_ref.p,mesh_ref.h,s = s[i])
    uhs,mhs  = FracMFG.Solve(H,dH,Fm,F,G,s = s[i],mu = 0,C_H = 1,
                            nonlocaloperator = 'FracLaplUniform', 
                            As = As_ref,p_ref = mesh_ref.p, func_ex = func_ex)
    FracMFG.ResetStiffnessMatrixLoadVector()
    
    e_u_L2 = IntegrateL2(lambda x : 0,uh-uhs,mesh, root = False)
    e_m_L2 = IntegrateL2(lambda x : 0,mh-mhs,mesh, root = False)
    
    e_u_H1 = IntegrateH12(uh-uhs,mesh, root = False) + e_u_L2
    e_m_H1 = IntegrateH12(mh-mhs,mesh, root = False) + e_m_L2

    e_u_max = np.max(np.abs(uh-uhs))
    e_m_max = np.max(np.abs(mh-mhs))
    
    err_u_L2[i] = np.sqrt(e_u_L2)
    err_u_H1[i] = np.sqrt(e_u_H1)
    err_m_L2[i] = np.sqrt(e_m_L2)
    err_m_H1[i] = np.sqrt(e_m_H1)
    err_u_max[i] = e_u_max
    err_m_max[i] = e_m_max


labelsize = 15

savefigure = True

plt.figure(figsize = (5,5))
plt.loglog(1-s,err_u_max, label = r'$\lVert u^s_k - u_k \rVert_{L^\infty}$', marker = 'o')
plt.loglog(1-s,err_m_max, label = r'$\lVert m^s_k - m_k \rVert_{L^\infty}$', marker = 'o')
plt.loglog(1-s, ((1-s)/(1-s[0]))*0.5*(err_u_max[0]+err_m_max[0]),'--', label = r'Linear line: $1-s$')
plt.gca().invert_xaxis()  # Optional: shows values approaching 1 from the left
plt.xlabel(r'$1 - s$', fontsize = labelsize)
plt.ylabel('Error',fontsize = labelsize)
plt.title(r'Continuity of solutions $(u_k^s,m_k^s)$ as $s\to 1$',fontsize = labelsize)
#plt.grid(True)
plt.legend(fontsize = labelsize-2)
plt.show()

plt.figure(figsize = (5,5))
plt.loglog(1-s,err_u_L2,marker = 'o', label = r'$\lVert u^s_k - u_k \rVert_{L^2}$')
plt.loglog(1-s,err_u_H1,marker = 'o', label = r'$\lVert u^s_k - u_k \rVert_{H^1}$')
plt.loglog(1-s,err_m_L2,marker = 'o', label = r'$\lVert m^s_k - m_k \rVert_{L^2}$')
plt.loglog(1-s,err_m_H1,marker = 'o', label = r'$\lVert m^s_k - m_k \rVert_{H^1}$')
plt.loglog(1-s, ((1-s)/(1-s[0]))*0.25*(err_u_L2[0]+err_u_H1[0]+err_m_L2[0]+err_m_H1[0]),'--', label = r'Linear line: $1-s$')
plt.gca().invert_xaxis()  # Optional: shows values approaching 1 from the left
plt.xlabel(r'$1 - s$', fontsize = labelsize)
plt.ylabel('Error', fontsize = labelsize)
plt.title(r'Continuity of solutions $(u_k^s,m_k^s)$ as $s\to 1$', fontsize = labelsize)
#plt.grid(True)
plt.legend(fontsize = labelsize-2)
plt.show()