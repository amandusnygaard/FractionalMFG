import numpy as np
from scipy.special import gamma, rgamma
from scipy.linalg import toeplitz

from AcuteMesh import Mesh1D
from FracMFG1DHelper import *

# basis functions on interval [0,1] for use in gaussian quadrature
phi = [lambda x : 1-x, lambda x : x]
phi_grad = [-1, 1]

class MFGSolver1D:

    def __init__(self,mesh,nu,kappa,eps):
        
        self.mesh = mesh
        self.N_p = mesh.N_p
        self.N_T = mesh.N_T

        self.h = mesh.h

        self.stiffmat = np.zeros((self.N_p-2,self.N_p-2))
        self.loadvec = np.zeros((2,self.N_p-2))

        self.nu = nu
        self.kappa = kappa
        self.eps = eps

        self.gamma = 0

        return None
    
    def ArtificalDiffusionUniform(self, mu, C_H):
        
        self.gamma = max(mu*(C_H+self.kappa*self.h[0])*self.h[0]/2-self.eps,0)

        return None
    
    def StiffnessLaplaceUniform(self):

        e = -np.ones(self.N_p-3)
        d = 2*np.ones(self.N_p-2)
        
        self.stiffmat += (self.eps+self.gamma)/self.h[0]*( np.diag(e,k = -1)
                                             +np.diag(d,k = 0)
                                             +np.diag(e,k = 1))
        
        return None
    
    def StiffnessConstantUniform(self):

        e = 1/6*np.ones(self.N_p-3)
        d = 2/3*np.ones(self.N_p-2)

        self.stiffmat += self.kappa*self.h[0]*( np.diag(e,k = -1)
                                               +np.diag(d,k = 0)
                                               +np.diag(e,k = 1))
        
        return None

    def NonLocalStiffnessMatrix(self,s,nonlocaloperator = 'FracLaplUniform'):

        if nonlocaloperator == 'FracLaplUniform':
            self.stiffmat += self.nu*StiffnessFractionalLaplaceUniform(self.mesh.p,self.mesh.h,s)
        
        return None
    
    def AssembleLinearization(self,dH,U):
        
        linmat = np.zeros((self.N_p-2,self.N_p-2))

        for i in range(1,self.N_T-1):

            dHU = dH((U[i+1]-U[i])/self.h[i])

            linmat[i-1,i-1:i+1] += -0.5*self.h[i]*dHU
            linmat[i,i-1:i+1] += 0.5*self.h[i]*dHU

        linmat[0,0] += 0.5*self.h[0]*dH((U[1])/self.h[0])

        linmat[self.N_p-3,self.N_p-3] += -0.5*self.h[-1]*dH((-U[-2])/self.h[-1])

        return linmat
    
    def LoadVector(self,f0, eq = 0):

        x0 = self.mesh.p[0]
        x1 = self.mesh.p[1]

        self.loadvec[eq,0] += self.h[0]*fquad(f0,1,x0,x1)

        for i in range(1,self.N_T-1):

            x0 = self.mesh.p[i]
            x1 = self.mesh.p[i+1]

            self.loadvec[eq,i-1] += self.h[i]*fquad(f0,0,x0,x1)
            self.loadvec[eq,i] += self.h[i]*fquad(f0,1,x0,x1)

        x0 = self.mesh.p[self.N_p-2]
        x1 = self.mesh.p[self.N_p-1]

        self.loadvec[eq,self.N_p-3] += self.h[-1]*fquad(f0,0,x0,x1)

        return None
    
    def LoadVectorGrad(self,f1, eq = 0):

        x0 = self.mesh.p[0]
        x1 = self.mesh.p[1]

        self.loadvec[eq,0] += self.h[0]*quad(f1,x0,x1)

        for i in range(1,self.N_T-1):

            x0 = self.mesh.p[i]
            x1 = self.mesh.p[i+1]

            self.loadvec[eq,i-1] += -self.h[i]*quad(f1,x0,x1)
            self.loadvec[eq,i] += self.h[i]*quad(f1,x0,x1)

        x0 = self.mesh.p[self.N_p-2]
        x1 = self.mesh.p[self.N_p-1]
        
        self.loadvec[eq,self.N_p-3] += -self.h[-1]*quad(f1,x0,x1)

        return None
    
    def NonLocalLoadVector(self,As,func_ex,p_ref):
        '''
        Exploit that the points in the refinement are added at the end of the new nodes array.
        Then if u_ex is the interpolation function of the highly refined mesh,
        then a_s(u_ex,phi_i) = U_ex.T@As@Phi_i, where U_ex is the is the interpolation coefficients
        on the refined mesh, or just evaluating u_ex at each node. The same holds for Phi_i
        '''
        phi_val = compute_phi_1d_sparse(self.mesh.p, p_ref)
        
        func0_val = func_ex[0](p_ref[1:-1])
        func1_val = func_ex[1](p_ref[1:-1])

        Asphi = As@phi_val[1:-1,1:-1]

        self.loadvec[0,:] += self.nu*func0_val@Asphi
        self.loadvec[1,:] += self.nu*func1_val@Asphi

        return None
    
    def SolveKFP(self,G,dH, U):

        self.StiffnessLaplaceUniform()
        self.StiffnessConstantUniform()

        self.LoadVector(G[0], eq = 1)
        if not (G[1] is None):
            self.LoadVectorGrad(G[1], eq = 1)

        linmat = self.AssembleLinearization(dH,U)
        
        mh = np.zeros(self.N_p)
        mh[1:-1] = np.linalg.solve(self.stiffmat+linmat,self.loadvec[1,:])

        return mh
    
    def CouplingTerm(self,Fm,M):

        couplvec = np.zeros(self.N_p-2)

        for i in range(1,self.N_T-1):
            
            couplvec[i-1] += self.h[i]*Fmquad(Fm,M[i:i+2],0)
            couplvec[i]   += self.h[i]*Fmquad(Fm,M[i:i+2],1)

        couplvec[0] += self.h[0]*Fmquad(Fm,M[:2],0)
        couplvec[self.N_p-3] += self.h[-1]*Fmquad(Fm,M[-2:],0)
        
        return couplvec
    
    def Residual(self,U,H):

        res = np.zeros(self.N_p-2)

        for i in range(1,self.N_T-1):

            res[i-1:i+1] += 0.5*self.h[i]*H((U[i+1]-U[i])/self.h[i])

        res[0] += 0.5*self.h[0]*H(U[1]/self.h[0])
        res[-1] += 0.5*self.h[-1]*H(-U[-2]/self.h[-1])

        res -= self.loadvec[0,:]
        res += self.stiffmat@U[1:-1]

        return res
    
    def SolveHJB(self,Fm,M,F,H,dH, maxiter = 30, tol = 1e-11, uh = None):

        self.StiffnessLaplaceUniform()
        self.StiffnessConstantUniform()

        self.LoadVector(F[0], eq = 0)
        if not (F[1] is None):
            self.LoadVectorGrad(F[1], eq = 0)

        couplvec = self.CouplingTerm(Fm,M)

        if uh is None:
            uh  = np.zeros(self.N_p)

        du = np.zeros(self.N_p-2)
        
        for _ in range(maxiter):

            linmat = self.AssembleLinearization(dH,uh)
            
            res = self.Residual(uh,H) - couplvec
            
            du = np.linalg.solve(self.stiffmat + linmat, -res)

            uh[1:-1] += du
            
            if np.linalg.norm(du)/np.linalg.norm(uh) < tol:
                break

        if _ == maxiter -1:
            print('Did not converge')

        return uh
    
    def SolveKFPStep(self,dH, U):

        linmat = self.AssembleLinearization(dH,U)
        
        mh = np.zeros(self.N_p)
        mh[1:-1] = np.linalg.solve(self.stiffmat+linmat,self.loadvec[1,:])

        return mh
    
    def SolveHJBStep(self,Fm,M,H,dH, maxiter = 30, tol = 1e-11, uh0 = None):
        
        couplvec = self.CouplingTerm(Fm,M)
        
        if uh0 is None:
            uh  = np.zeros(self.N_p)
        else:
            uh = np.copy(uh0)

        du = np.zeros(self.N_p-2)
        
        for _ in range(maxiter):

            linmat = self.AssembleLinearization(dH,uh)
            
            res = self.Residual(uh,H) - couplvec
            
            du = np.linalg.solve(self.stiffmat + linmat, -res)
            
            uh[1:-1] += du
            
            if np.linalg.norm(du)/np.linalg.norm(uh) < tol:
                break

        if _ == maxiter -1:
            print('Did not converge')

        return uh

    def Solve(self,H,dH,Fm,F,G,s = None,mu = 1.0001, C_H = 1,nonlocaloperator = 'FracLaplUniform',
              As = None,p_ref = None,func_ex = None,
              maxiter_policy = 10, tol_policy = 1e-8, maxiter_newton = 10, tol_newton = 1e-11,
              check_stabilization = False, beta = 1):
        
        uh0 = np.zeros(self.N_p)
        mh0 = np.zeros(self.N_p)
        uh = np.zeros(self.N_p)
        mh = np.zeros(self.N_p)

        self.ArtificalDiffusionUniform(mu,C_H)

        # Compute all static matrices/vectors
        self.StiffnessLaplaceUniform()
        self.StiffnessConstantUniform()
        if not (nonlocaloperator is None or s is None):
            self.NonLocalStiffnessMatrix(s,nonlocaloperator)

        self.LoadVector(F, eq = 0)
        
        self.LoadVector(G[0], eq = 1)
        if not (G[1] is None):
            self.LoadVectorGrad(G[1], eq = 1)

        if not (nonlocaloperator is None or s is None or func_ex is None):
            self.NonLocalLoadVector(As,func_ex,p_ref)

        # Solve KFP with u = 0 to get a start value
        mh0[1:-1] = np.linalg.solve(self.stiffmat,self.loadvec[1,:])
        
        # Solve HJB with the computed mh
        uh0[:] = self.SolveHJBStep(Fm,mh0,H,dH,maxiter_newton,tol_newton)
        
        # Now perform the policy iteration
        for _ in range(maxiter_policy):

            if beta == 1:
                mh[:] = self.SolveKFPStep(dH,uh0)
                uh[:] = self.SolveHJBStep(Fm,mh,H,dH,maxiter_newton,tol_newton, uh0 = uh0)
    
            else:
                mh[:] = (1-beta)*mh0[:] + beta*self.SolveKFPStep(dH,uh0)
                uh[:] = (1-beta)*uh0[:] + beta*self.SolveHJBStep(Fm,mh,H,dH,maxiter_newton,tol_newton, uh0 = uh0)

            #print(np.linalg.norm(uh-uh0))
            if max(np.linalg.norm(uh-uh0)/np.linalg.norm(uh),
                   np.linalg.norm(mh-mh0)/np.linalg.norm(mh)) < tol_policy:
                break
            
            mh0[:] = mh[:]
            uh0[:] = uh[:]

        if _ == maxiter_policy-1:
            print('Policy Did not converge!!!!')
        
        if check_stabilization:
            return uh,mh, (not np.all(self.gamma == 0))

        return uh,mh
    
    def ResetStiffnessMatrixLoadVector(self):
        self.stiffmat = np.zeros((self.N_p-2,self.N_p-2))
        self.loadvec = np.zeros((2,self.N_p-2))

        return None

    
def ConvergenceTest(mesh,
                    eps,
                    nu,
                    kappa,
                    H,
                    dH,
                    Fm,
                    F: list,
                    G: list,
                    sol: list,
                    diff_sol: list,
                    sol_norms: list = None,
                    mu = 1.00001,
                    C_H = 1,
                    num_refs: int = 4,
                    As = None,
                    s = None,
                    p_ref = None,
                    start_ref = 0,
                    filename = None,
                    ):
    
    
    h_list = np.zeros(num_refs+1)

    err_u_L2 = np.zeros(num_refs+1)
    err_u_H1 = np.zeros(num_refs+1)
    err_m_L2 = np.zeros(num_refs+1)
    err_m_H1 = np.zeros(num_refs+1)
    stabilization = np.zeros(num_refs+1)

    mesh.Refine(start_ref)
    h_list[0] = np.max(mesh.h)

    for i in range(num_refs+1):

        if i > 0:
            mesh.Refine()
            h_list[i] = np.max(mesh.h)
        
        FracMFG = MFGSolver1D(mesh,nu,kappa,eps)

        uh,mh,stabilization[i]  = FracMFG.Solve(H,dH,Fm,F,G,s = s,mu = mu,C_H = C_H,
                                                nonlocaloperator = 'FracLaplUniform', 
                                                As = As,p_ref = p_ref, func_ex = sol,
                                                check_stabilization=True)

        e_u_L2 = IntegrateL2(sol[0],uh,mesh, root = False)
        e_m_L2 = IntegrateL2(sol[1],mh,mesh, root = False)

        e_u_H1 = IntegrateH1(diff_sol[0],uh,mesh, root = False) + e_u_L2
        e_m_H1 = IntegrateH1(diff_sol[1],mh,mesh, root = False) + e_m_L2
        
        if not (sol_norms is None):
            err_u_L2[i] = np.sqrt(e_u_L2)/sol_norms[0]
            err_u_H1[i] = np.sqrt(e_u_H1)/sol_norms[1]
            err_m_L2[i] = np.sqrt(e_m_L2)/sol_norms[2]
            err_m_H1[i] = np.sqrt(e_m_H1)/sol_norms[3]
        else:
            err_u_L2[i] = np.sqrt(e_u_L2)
            err_u_H1[i] = np.sqrt(e_u_H1)
            err_m_L2[i] = np.sqrt(e_m_L2)
            err_m_H1[i] = np.sqrt(e_m_H1)

    if not (filename is None):
        np.savez(filename,
                 err_u_L2= err_u_L2,
                 err_u_H1= err_u_H1,
                 err_m_L2= err_m_L2,
                 err_m_H1 = err_m_H1, 
                 h_list = h_list,
                 stabilization = stabilization)


    return err_u_L2,err_u_H1,err_m_L2,err_m_H1, h_list,stabilization
