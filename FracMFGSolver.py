import numpy as np
import scipy.io as sio

from AcuteMesh import Mesh
from FracMFGHelper import *
from FracLaplace import FractionalLaplaceStiffMat,RegionalFractionalLaplaceStiffMat

phi = [lambda x : 1-x[0]-x[1], lambda x: x[0], lambda x : x[1]]
phi_grad = [np.array([-1,-1]),np.array([1,0]),np.array([0,1])]


class HelmHoltzSolver:

    def __init__(self, mesh):
        
        self.mesh = mesh
        self.N_p = mesh.N_p
        self.N_T = mesh.N_T

        self.mat = np.zeros(shape = (self.N_p,self.N_p))

        self.vec = np.zeros(self.N_p)

        return None


    def LocalStiffnessMatrixLaplace(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True)

        c = area
        
        for j in range(3):
            self.mat[T[j],T[j]] += c*np.inner(K.T@phi_grad[j],K.T@phi_grad[j])

            for k in range(j):
                value = c*np.inner(K.T@phi_grad[k],K.T@phi_grad[j])
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value

        return None

    def LocalStiffnessMatrixLowestOrder(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        y0 = np.array([0,0.5])
        y1 = np.array([0.5,0])
        y2 = np.array([0.5,0.5])

        c = (2*area)/6

        for j in range(3):

            self.mat[T[j],T[j]] += c*(  phi[j](y0)**2 
                                      + phi[j](y1)**2 
                                      + phi[j](y2)**2)
            for k in range(j):
                value = c*(  phi[j](y0)*phi[k](y0)
                           + phi[j](y1)*phi[k](y1)
                           + phi[j](y2)*phi[k](y2))
                
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value


        return None
    
    def StiffnessMatrix(self):

        for i in range(self.N_T):
            self.LocalStiffnessMatrixLaplace(i)
            self.LocalStiffnessMatrixLowestOrder(i)

        return None


    def LocalLoadVector(self,f,i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = False)

        c = 2*area

        A = lambda x: K@x + x0

        for j in range(3):

            self.vec[T[j]] += c*Quad(f,A,j)

        return None


    def LoadVector(self,f):

        for i in range(self.N_T):
            self.LocalLoadVector(f,i)
        
        return None

    def TrimMatVec(self):

        self.mat =  self.mat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]
        self.vec = self.vec[~self.mesh.bndrynode]

    
    def Solve(self,f):

        self.StiffnessMatrix()
        self.LoadVector(f)
        
        self.TrimMatVec()
        
        uh = np.zeros(self.N_p)

        uh[~self.mesh.bndrynode] += np.linalg.solve(self.mat, self.vec)

        return uh


class KFPSolver:

    def __init__(self, mesh):
        
        self.mesh = mesh
        self.N_p = mesh.N_p
        self.N_T = mesh.N_T

        self.mat = np.zeros(shape = (self.N_p,self.N_p))

        self.vec = np.zeros(self.N_p)

        return None


    def LocalStiffnessMatrixLaplace(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True)

        c = area
        
        for j in range(3):
            self.mat[T[j],T[j]] += c*np.inner(K.T@phi_grad[j],K.T@phi_grad[j])

            for k in range(j):
                value = c*np.inner(K.T@phi_grad[k],K.T@phi_grad[j])
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value

        return None

    def LocalStiffnessMatrixLowestOrder(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        y0 = np.array([0,0.5])
        y1 = np.array([0.5,0])
        y2 = np.array([0.5,0.5])

        c = (2*area)/6

        for j in range(3):

            self.mat[T[j],T[j]] += c*(  phi[j](y0)**2 
                                      + phi[j](y1)**2 
                                      + phi[j](y2)**2)
            for k in range(j):
                value = c*(  phi[j](y0)*phi[k](y0)
                           + phi[j](y1)*phi[k](y1)
                           + phi[j](y2)*phi[k](y2))
                
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value


        return None
    
    def LocalStiffnessMatrixHamiltonian(self,dH,U,i):
        '''
        dH : gradient of non-linear Hamiltonian
        U : a vector of gridfunction coefficients
        '''

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        c = 2*area

        K = ReferenceTriangleTransformJacobian(x0,x1,x2)

        dHU = dH(K.T@(U[T[0]]*phi_grad[0] + U[T[1]]*phi_grad[1]+U[T[2]]*phi_grad[2]))

        for j in range(3):
            for k in range(3):

                self.mat[T[j],T[k]] += c/6*np.inner(dHU,K.T@phi_grad[j])

        return None
    
    def LocalStiffnessMatrixHamiltonianTest(self,dHU,i):
        '''
        dH : gradient of non-linear Hamiltonian
        U : a vector of gridfunction coefficients
        '''

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        c = 2*area

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse=False)
        Kinv = np.linalg.inv(K)

        A = lambda x : K@x+x0

        for j in range(3):
            for k in range(3):

                self.mat[T[k],T[j]] += c*np.inner(QuadGrad(dHU,A,j),Kinv.T@phi_grad[k])

        return None
    
    def StiffnessMatrix(self, dH, U):

        for i in range(self.N_T):
            self.LocalStiffnessMatrixLaplace(i)
            self.LocalStiffnessMatrixLowestOrder(i)
            self.LocalStiffnessMatrixHamiltonian(dH,U,i)

        return None
    
    def StiffnessMatrixTest(self, dHU):

        for i in range(self.N_T):
            self.LocalStiffnessMatrixLaplace(i)
            self.LocalStiffnessMatrixLowestOrder(i)
            self.LocalStiffnessMatrixHamiltonianTest(dHU,i)

        return None


    def LocalLoadVector(self,i,G0,G1 = None):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = False)

        c = 2*area

        A = lambda x: K@x + x0

        for j in range(3):

            self.vec[T[j]] += c*Quad(G0,A,j)

        if not (G1 is None):

            Kinv = np.linalg.inv(K)

            for j in range(3):

                self.vec[T[j]] += c*np.inner(QuadGrad(G1,A),Kinv.T@phi_grad[j])

        return None


    def LoadVector(self,G0,G1 = None):

        for i in range(self.N_T):
            self.LocalLoadVector(i,G0,G1)
        
        return None

    def TrimMatVec(self):

        self.mat =  self.mat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]
        self.vec = self.vec[~self.mesh.bndrynode]

    def Solve(self,dH,U,G0,G1 = None):

        self.StiffnessMatrix(dH,U)
        self.LoadVector(G0,G1)

        self.TrimMatVec()

        mh = np.zeros(self.N_p)

        mh[~self.mesh.bndrynode] += np.linalg.solve(self.mat,self.vec)

        return mh

    def SolveTest(self,dHU,G0,G1 = None):

        self.StiffnessMatrixTest(dHU)
        self.LoadVector(G0,G1)

        self.TrimMatVec()

        mh = np.zeros(self.N_p)

        mh[~self.mesh.bndrynode] += np.linalg.solve(self.mat,self.vec)

        return mh
    

class HJBSolver:

    def __init__(self, mesh):
        
        self.mesh = mesh
        self.N_p = mesh.N_p
        self.N_T = mesh.N_T

        self.mat = np.zeros(shape = (self.N_p,self.N_p))

        self.vec = np.zeros(self.N_p)

        return None


    def LocalStiffnessMatrixLaplace(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True)

        c = area
        
        for j in range(3):
            self.mat[T[j],T[j]] += c*np.inner(K.T@phi_grad[j],K.T@phi_grad[j])

            for k in range(j):
                value = c*np.inner(K.T@phi_grad[k],K.T@phi_grad[j])
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value

        return None

    def LocalStiffnessMatrixLowestOrder(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        y0 = np.array([0,0.5])
        y1 = np.array([0.5,0])
        y2 = np.array([0.5,0.5])

        c = (2*area)/6

        for j in range(3):

            self.mat[T[j],T[j]] += c*(  phi[j](y0)**2 
                                      + phi[j](y1)**2 
                                      + phi[j](y2)**2)
            for k in range(j):
                value = c*(  phi[j](y0)*phi[k](y0)
                           + phi[j](y1)*phi[k](y1)
                           + phi[j](y2)*phi[k](y2))
                
                self.mat[T[k],T[j]] += value

                self.mat[T[j],T[k]] += value


        return None

    def StiffnessMatrix(self):

        for i in range(self.N_T):
            self.LocalStiffnessMatrixLaplace(i)
            self.LocalStiffnessMatrixLowestOrder(i)

        return None
    
    def LocalAssembleLinearization(self,dH,U,i):
        '''
        dH : gradient of non-linear Hamiltonian
        U : a vector of gridfunction coefficients
        '''

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        c = 2*area

        K = ReferenceTriangleTransformJacobian(x0,x1,x2)

        dHU = dH(K.T@(U[T[0]]*phi_grad[0] + U[T[1]]*phi_grad[1]+U[T[2]]*phi_grad[2]))

        for j in range(3):

            self.linmat[T[j],T] += c/6*np.inner(dHU,K.T@phi_grad[j])


        return None

    
    def AssembleLinearization(self,dH,U):

        self.linmat = np.zeros(shape = (self.N_p,self.N_p))

        for i in range(self.N_T):

            self.LocalAssembleLinearization(dH,U,i)

        self.linmat =  self.linmat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]

        return None


    def LocalLoadVector(self,i,G0,G1 = None):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = False)

        c = 2*area

        A = lambda x: K@x + x0

        for j in range(3):

            self.vec[T[j]] += c*Quad(G0,A,j)

        if not (G1 is None):

            Kinv = np.linalg.inv(K)

            for j in range(3):

                self.vec[T[j]] += c*np.inner(QuadGrad(G1,A),Kinv.T@phi_grad[j])

        return None


    def LoadVector(self,F0,F1 = None):

        for i in range(self.N_T):
            self.LocalLoadVector(i,F0,F1)
        
        return None


    def LocalHamiltonTerm(self,i,U,H):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]
        x1 = self.mesh.p[:,T[1]]
        x2 = self.mesh.p[:,T[2]]

        c = 2*area

        K = ReferenceTriangleTransformJacobian(x0,x1,x2)

        HU = H(K.T@(U[T[0]]*phi_grad[0] + U[T[1]]*phi_grad[1]+U[T[2]]*phi_grad[2]))

        self.res[T] += c/6*HU

        return None

    def Residual(self,U,H):

        self.res = np.zeros(self.N_p)

        for i in range(self.N_T):

            self.LocalHamiltonTerm(i,U,H)

        self.res = self.res[~self.mesh.bndrynode]

        self.res -= self.vec
        self.res += self.mat@U[~self.mesh.bndrynode]

        return None

    def TrimMatVec(self):

        self.mat =  self.mat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]
        self.vec = self.vec[~self.mesh.bndrynode]

        return None

    def Solve(self,H,dH,F0,F1 = None, maxiter = 30, tol = 1e-11):

        self.StiffnessMatrix()
        self.LoadVector(F0,F1)

        self.TrimMatVec()

        uh = np.zeros(self.N_p)
        du = np.zeros(self.N_p)

        for _ in range(maxiter):

            self.AssembleLinearization(dH,uh)
            self.Residual(uh,H)

            du[~self.mesh.bndrynode] = np.linalg.solve(self.mat + self.linmat, -self.res)

            uh += du

            if np.linalg.norm(du) < tol:
                break

        return uh


class MFGSolver:

    def __init__(self, mesh, eps,nu, kappa):
        
        self.mesh = mesh
        self.N_p = mesh.N_p
        self.N_T = mesh.N_T
        self.N_e = mesh.N_e

        self.stiffmat = np.zeros(shape = (self.N_p,self.N_p))

        # vector containing both load vectors for the HJB equation and KFP equation,
        # which do not change in the policy iteration.
        # loadvec[0,:] gives static load vector for HJB equation.
        self.loadvec = np.zeros(shape = (2,self.N_p))

        self.eps = eps
        self.kappa = kappa
        self.nu = nu

        self.gamma = np.zeros(self.N_T)

        # matrix containing all matrices associated with the affine transform to reference integral
        # and its inverses.
        # Matric JT[:,2i:2i+2] gives the matrix for element i.
        self.JT = np.zeros((2,2*self.N_T))
        self.JTinv = np.zeros((2,2*self.N_T))

        return None
    
    def AffineTransformMatrix(self):
        
        for i in range(self.N_T):

            T = self.mesh.t[i]

            x0 = self.mesh.p[:,T[0]]
            x1 = self.mesh.p[:,T[1]]
            x2 = self.mesh.p[:,T[2]]

            self.JT[:,2*i:2*i+2] = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse=False)
            self.JTinv[:,2*i:2*i+2] = np.linalg.inv(self.JT[:,2*i:2*i+2])
    
        return None

    def ArtificialDiffusion(self,mu, C_H):
        
        if self.mesh.theta is None:
            self.mesh.DiamSizeAngleGlobal(find_angle = True)

        # Find sigma
        sigma = np.inf

        for i in range(self.N_T):

            T = self.mesh.t[i]

            # x0 = self.mesh.p[:,T[0]]
            # x1 = self.mesh.p[:,T[1]]
            # x2 = self.mesh.p[:,T[2]]

            #K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True)
            Kinv = self.JTinv[:,2*i:2*i+2]

            sigmaT0 = np.linalg.norm(Kinv.T@phi_grad[0])
            sigmaT1 = np.linalg.norm(Kinv.T@phi_grad[1])
            sigmaT2 = np.linalg.norm(Kinv.T@phi_grad[2])

            sigmaT = min(sigmaT0,sigmaT1,sigmaT2)

            sigma = min(sigma,self.mesh.h[i]*sigmaT)

        sintheta = np.sin(0.5*np.pi-self.mesh.theta)

        for i in range(self.N_T):
            self.gamma[i] = max(mu*(C_H+self.kappa*self.mesh.h[i])*self.mesh.h[i]/(sigma*sintheta)
                                -self.eps,0)
        
        return None
    
    def LocalStiffnessMatrixLaplace(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        # x0 = self.mesh.p[:,T[0]]
        # x1 = self.mesh.p[:,T[1]]
        # x2 = self.mesh.p[:,T[2]]

        #K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = True)
        Kinv = self.JTinv[:,2*i:2*i+2]

        c = (self.eps+ self.gamma[i])*area
        
        for j in range(3):
            self.stiffmat[T[j],T[j]] += c*np.inner(Kinv.T@phi_grad[j],Kinv.T@phi_grad[j])

            for k in range(j):
                value = c*np.inner(Kinv.T@phi_grad[k],Kinv.T@phi_grad[j])
                self.stiffmat[T[k],T[j]] += value

                self.stiffmat[T[j],T[k]] += value

        return None
    
    def LocalStiffnessMatrixLowestOrder(self, i):

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        y0 = np.array([0,0.5])
        y1 = np.array([0.5,0])
        y2 = np.array([0.5,0.5])

        c = self.kappa*(2*area)/6

        for j in range(3):

            self.stiffmat[T[j],T[j]] += c*(  phi[j](y0)**2 
                                      + phi[j](y1)**2 
                                      + phi[j](y2)**2)
            for k in range(j):
                value = c*(  phi[j](y0)*phi[k](y0)
                           + phi[j](y1)*phi[k](y1)
                           + phi[j](y2)*phi[k](y2))
                
                self.stiffmat[T[k],T[j]] += value

                self.stiffmat[T[j],T[k]] += value


        return None
    
    def NonLocalStiffnessMatrix(self,s, nonlocaloperator = 'FracLapl'):

        
        if nonlocaloperator == 'FracLapl':

            self.stiffmat += self.nu*FractionalLaplaceStiffMat(self.mesh.p,self.mesh.t,self.mesh.e,s,
                                                               self.N_p,self.N_T,self.N_e,
                                                               self.mesh.area,self.mesh.length,
                                                               self.mesh.bndrynode,self.mesh.normals,
                                                               self.mesh.patches,self.mesh.edgepatches)

        
        if nonlocaloperator == 'RegFracLapl':

            self.stiffmat += self.nu*RegionalFractionalLaplaceStiffMat(self.mesh.p,self.mesh.t,s,
                                                                       self.N_p,self.N_T,
                                                                       self.mesh.area,
                                                                       self.mesh.bndrynode,
                                                                       self.mesh.patches,)

        return None
    
    def StiffnessMatrix(self):

        if self.kappa == 0:

            for i in range(self.N_T):
                self.LocalStiffnessMatrixLaplace(i)
            
            return None

        for i in range(self.N_T):
            self.LocalStiffnessMatrixLaplace(i)
            self.LocalStiffnessMatrixLowestOrder(i)

        return None

    def LocalLoadVector(self,i,f0,f1 = None, eq = 0):
        '''
        eq = 0,1 : for which equation is the static load vector computed
        '''

        T = self.mesh.t[i]
        area = self.mesh.area[i]

        x0 = self.mesh.p[:,T[0]]

        #K = ReferenceTriangleTransformJacobian(x0,x1,x2, inverse = False)
        K = self.JT[:,2*i:2*i+2]

        c = 2*area

        A = lambda x: K@x + x0

        for j in range(3):

            self.loadvec[eq,T[j]] += c*Quad(f0,A,j)

        if not (f1 is None):

            Kinv = np.linalg.inv(K)

            for j in range(3):

                self.loadvec[eq,T[j]] += c*np.inner(QuadGrad(f1,A),Kinv.T@phi_grad[j])

        return None


    def LoadVector(self,f0,f1 = None, eq = 0):

        for i in range(self.N_T):
            self.LocalLoadVector(i,f0,f1,eq)
        
        return None
    
    def NonLocalLoadVector(self,As,func_ex,p_ref):
        '''
        Exploit that the points in the refinement are added at the end of the new nodes array.
        Then if u_ex is the interpolation function of the highly refined mesh,
        then a_s(u_ex,phi_i) = U_ex.T@As@Phi_i, where U_ex is the is the interpolation coefficients
        on the refined mesh, or just evaluating u_ex at each node. The same holds for Phi_i
        '''

        phi_val = compute_phi_on_refined_mesh_sparse(self.mesh.p,self.mesh.t,p_ref)
        
        func0_val = func_ex[0](p_ref)
        func1_val = func_ex[1](p_ref)

        Asphi = As@phi_val

        self.loadvec[0,:] += self.nu*func0_val@Asphi
        self.loadvec[1,:] += self.nu*func1_val@Asphi

        return None

    
    def TrimMatVec(self):

        self.stiffmat = self.stiffmat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]
        self.loadvec = self.loadvec[:,~self.mesh.bndrynode]

        return None
    
    
    def CouplingTerm(self,Fm,M):

        couplvec = np.zeros(self.N_p)

        for i in range(self.N_T):

            T = self.mesh.t[i]
            area = self.mesh.area[i]

            y0 = np.array([0,0.5])
            y1 = np.array([0.5,0])
            y2 = np.array([0.5,0.5])

            Mloc = M[T]

            c = 2*area/6

            for j in range(3):

                couplvec[T[j]] += c*(  Fm(Mloc[0]*phi[0](y0)+Mloc[1]*phi[1](y0)+Mloc[2]*phi[2](y0))*phi[j](y0)
                                     + Fm(Mloc[0]*phi[0](y1)+Mloc[1]*phi[1](y1)+Mloc[2]*phi[2](y1))*phi[j](y1)
                                     + Fm(Mloc[0]*phi[0](y2)+Mloc[1]*phi[1](y2)+Mloc[2]*phi[2](y2))*phi[j](y2))

        return couplvec[~self.mesh.bndrynode]
    
    def AssembleLinearization(self,dH,U):

        linmat = np.zeros(shape = (self.N_p,self.N_p))

        for i in range(self.N_T):

            T = self.mesh.t[i]
            area = self.mesh.area[i]

            c = 2*area

            #K = ReferenceTriangleTransformJacobian(x0,x1,x2)
            Kinv = self.JTinv[:,2*i:2*i+2]

            dHU = dH(Kinv.T@(U[T[0]]*phi_grad[0] + U[T[1]]*phi_grad[1]+U[T[2]]*phi_grad[2]))

            for j in range(3):

                linmat[T[j],T] += c/6*np.inner(dHU,Kinv.T@phi_grad[j])

        return linmat[~self.mesh.bndrynode,:][:,~self.mesh.bndrynode]
    
    def Residual(self,U,H):
        '''
        Finds the residual without the F[m] part
        '''

        res = np.zeros(self.N_p)

        for i in range(self.N_T):

            T = self.mesh.t[i]
            area = self.mesh.area[i]

            c = 2*area

            #K = ReferenceTriangleTransformJacobian(x0,x1,x2)
            Kinv = self.JTinv[:,2*i:2*i+2]

            HU = H(Kinv.T@(U[T[0]]*phi_grad[0] + U[T[1]]*phi_grad[1]+U[T[2]]*phi_grad[2]))

            res[T] += c/6*HU

        res = res[~self.mesh.bndrynode]

        res -= self.loadvec[0,:]
        res += self.stiffmat@U[~self.mesh.bndrynode]

        return res
    
    def HJBSolve(self,H,dH,Fm,M, maxiter = 30, tol = 1e-11, uh0 = None):

        couplevec = self.CouplingTerm(Fm,M)

        if uh0 is None:
            uh = np.zeros(self.N_p)
        else:
            uh = np.copy(uh0)

        du = np.zeros(self.N_p)

        for _ in range(maxiter):

            linmat = self.AssembleLinearization(dH,uh)
            res = self.Residual(uh,H) - couplevec

            du[~self.mesh.bndrynode] = np.linalg.solve(self.stiffmat + linmat, -res)

            uh += du

            if np.linalg.norm(du) < tol:
                break

        if _ == maxiter-1:
            print('Did not converge!')

        return uh
    
    def KFPSolve(self, dH,U = None):
        
        linmat = self.AssembleLinearization(dH,U)

        mh = np.zeros(self.N_p)

        mh[~self.mesh.bndrynode] += np.linalg.solve(self.stiffmat + linmat,self.loadvec[1,:])

        return mh
    
    def Solve(self,H,dH,Fm,F,G,s = None, mu = 1.1, C_H = 1, nonlocaloperator = 'FracLapl',
              As = None, p_ref = None, func_ex = None,
              maxiter_policy = 10, tol_policy = 1e-8, maxiter_newton = 30, tol_newton = 1e-11,
              check_stabilization = False):
        
        self.AffineTransformMatrix()

        uh0 = np.zeros(self.N_p)
        mh0 = np.zeros(self.N_p)
        uh = np.zeros(self.N_p)
        mh = np.zeros(self.N_p)

        self.ArtificialDiffusion(mu,C_H)
        
        # Compute all static matrices/vectors
        self.StiffnessMatrix()

        self.LoadVector(F[0],F[1], eq = 0)
        self.LoadVector(G[0],G[1], eq = 1)
        if not (s is None):
            
            self.NonLocalStiffnessMatrix(s,nonlocaloperator)

            self.NonLocalLoadVector(As,func_ex,p_ref)

        self.TrimMatVec()

        # Solve KFP with u = 0 to get a start value
        mh0[~self.mesh.bndrynode] = np.linalg.solve(self.stiffmat,self.loadvec[1,:])

        # Solve HJB with the computed mh
        uh0[:] = self.HJBSolve(H,dH,Fm,mh0,maxiter_newton,tol_newton)

        # Now perform the policy iteration
        for _ in range(maxiter_policy):

            mh[:] = self.KFPSolve(dH,uh0)
            uh[:] = self.HJBSolve(H,dH,Fm,mh,maxiter_newton,tol_newton, uh0 = uh0)

            if max(np.linalg.norm(uh-uh0),np.linalg.norm(mh-mh0)) < tol_policy:
                break

            mh0[:] = mh[:]
            uh0[:] = uh[:]

        if check_stabilization:
            return uh,mh, (not np.all(self.gamma == 0))

        return uh,mh


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
    mesh.ComputeSizes()
    h_list[0] = np.max(mesh.h)
    mesh.FindPatches()

    for i in range(num_refs+1):

        if i > 0:
            mesh.Refine()
            mesh.ComputeSizes()
            h_list[i] = np.max(mesh.h)
            mesh.FindPatches()
        
        FracMFG = MFGSolver(mesh,eps,nu,kappa)

        uh,mh,stabilization[i]  = FracMFG.Solve(H,dH,Fm,F,G,s = s,mu = mu,C_H = C_H,
                                                nonlocaloperator = 'FracLapl', 
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