
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma,rgamma
import matplotlib.pyplot as plt

import scipy.io as sio
from FracLaplaceHelper import *

data = sio.loadmat('data.mat')
phiA, phiB, phiD = (data.get(key) for key in ['phiA','phiB','phiD'])
tpsi1, tpsi2, tpsi3 = (data.get(key) for key in ['tpsi1', 'tpsi2', 'tpsi3'])
epsi1,epsi2,epsi3,epsi4,epsi5 = (data.get(key) for key in ['epsi1','epsi2','epsi3','epsi4','epsi5'])
vpsi1, vpsi2 = (data.get(key) for key in ['vpsi1', 'vpsi2'])
phi_edge = data.get("phi_edge")
epsi1_edge, epsi2_edge, epsi3_edge = (data.get(key) for key in ['epsi1_edge', 'epsi2_edge', 'epsi3_edge'])
epsi1_edge, epsi2_edge, epsi3_edge = epsi1_edge.flatten(), epsi2_edge.flatten(), epsi3_edge.flatten()
vpsi1_edge, vpsi2_edge = (data.get(key) for key in ['vpsi1_edge', 'vpsi2_edge'])

def FractionalLaplaceStiffMat(p,t,e,s,
                              N_p,N_T,N_e,
                              area,length,
                              bndrynode,normals,
                              patches,edgepatches):

    B = np.zeros((N_p,N_p))

    cns = 2**(2*s-1)*s*gamma(s+1)*rgamma(1-s)/np.pi

    vl = np.zeros((6,2))
    vm = np.zeros((6*N_T,2))
    ve = np.zeros((6,2))
    norms = np.zeros((36,N_T))

    ML = np.zeros((6,6,N_T))
    empty = np.zeros(N_T, dtype = np.int32)
    aux_ind = np.reshape(np.tile(np.arange(0, 3*N_T, 3), (6, 1)), -1)
    empty_vtx = np.zeros((2,3*N_T))
    BBm = np.zeros((2,2*N_T))

    edge_number_array = np.arange(N_e)
    for l in range(N_T):
        edge = np.concatenate((patches[t[l,0]],
                            patches[t[l,1]],
                            patches[t[l,2]]))
        
        nonempty, N = np.unique(edge, return_index=True)
        
        edge = np.delete(edge,N)

        vertex = np.setdiff1d(nonempty,edge)

        ll = N_T-l-np.sum(nonempty >= l)

        edge = edge[edge > l]
        vertex = vertex[vertex > l]

        empty[:ll] = np.setdiff1d(np.arange(l,N_T), nonempty)[:ll]

        empty_vtx[:,:3*ll] = p[:,t[empty[:ll],:].flatten()]
        
        Tl = t[l]

        xl = p[0,Tl]
        yl = p[1,Tl]

        K = np.array(((xl[1]-xl[0],xl[2]-xl[1]),(yl[1]-yl[0],yl[2]-yl[1])))
        
        #Identiacl elements
        B[Tl[:,None],Tl] += triangle_quad(K,s,tpsi1,tpsi2,tpsi3,area[l],p_I)
        
        #Disjoint elements
        BBm[0,:2*ll:2] = empty_vtx[0,1:3*ll:3]-empty_vtx[0,:3*ll:3]
        BBm[1,:2*ll:2] = empty_vtx[0,2:3*ll:3]-empty_vtx[0,1:3*ll:3]
        BBm[0,1:2*ll:2] = empty_vtx[1,1:3*ll:3]-empty_vtx[1,:3*ll:3]
        BBm[1,1:2*ll:2] = empty_vtx[1,2:3*ll:3]-empty_vtx[1,1:3*ll:3]
        
        vl = p_T_6 @ K.T + np.vstack([np.ones(6) * xl[0], np.ones(6) * yl[0]]).T
        for r in range(ll):
            vm[6*r:6*(r+1),:] = p_T_6@BBm[:,2*r:2*r+2] + empty_vtx[:,aux_ind[r]].T
        
        norms[:,:ll] = np.reshape(cdist(vl,vm[:6*ll, :]), (36,-1), order = 'F')**(-2-2*s)
        
        ML[:3,:3,:ll] = np.reshape(phiA@norms[:,:ll], (3,3,-1),order = 'F')
        ML[:3,3:6,:ll] = np.reshape(phiB@norms[:,:ll], (3,3,-1),order = 'F')
        ML[3:6,3:6,:ll] = np.reshape(phiD@norms[:,:ll], (3,3,-1),order = 'F')
        ML[3:6,:3,:ll] = np.transpose(ML[:3,3:6,:ll], (1,0,2))

        for m in range(ll):
            order = np.hstack((Tl,t[empty[m],:].flatten()))
            B[order[:,None],order] += 8*area[l]*area[empty[m]] * ML[:6,:6,m]


        #Vertex touching elements
        for m in vertex:
            Tm = t[m]
            T_com = np.intersect1d(Tl,Tm)
            order = np.hstack((T_com,Tl[Tl != T_com],Tm[Tm != T_com]))
            B[order[:,None],order] += 2*vertex_quad(Tl,Tm,T_com,p,s,vpsi1,vpsi2,
                                                    area[l],area[m],p_cube)

        #Edge touching elements
        for m in edge:
            Tm = t[m]
            T_diff = np.hstack([np.setdiff1d(Tl,Tm), np.setdiff1d(Tm,Tl)])
            order = np.hstack((Tl[Tl != T_diff[0]],T_diff))
            B[order[:,None],order] += 2*edge_quad(Tl,Tm,T_diff,p,s,epsi1,epsi2,epsi3,epsi4,epsi5,
                                            area[l],area[m],p_cube)
            
        #Boundary contributions
        on_bndry = bndrynode[Tl]
        num_on_bndry = np.sum(on_bndry)
        
        if num_on_bndry == 0:
            
            # All edges do not intersect
            for m in range(N_e):

                pe = p[:,e[m]]
                
                ve = (pe[:,1]-pe[:,0])[None,:]*p_I_6[:,None] + pe[:,0]

                pos_diff, pos_distsquare = pairwise_distances_with_vectors(vl,ve)
                
                d = np.multiply(normals[m]@pos_diff,pos_distsquare**(-1-s))
                B[Tl[:,None],Tl] += 2*area[l]*length[m]/s*np.reshape(phi_edge@d, (3,3), order = 'F')
        

        elif num_on_bndry == 1:
            
            idx_on_bndry = Tl[on_bndry][0]

            vertex_edges = edgepatches[idx_on_bndry]


            non_intersect_edge = np.setdiff1d(edge_number_array,vertex_edges)

            #Non intersecting edges
            for m in non_intersect_edge:

                pe = p[:,e[m]]
                ve = (pe[:,1]-pe[:,0])[None,:]*p_I_6[:,None] + pe[:,0]

                pos_diff, pos_diff_dist = pairwise_distances_with_vectors(vl,ve)

                d = np.multiply(normals[m]@pos_diff,pos_diff_dist**(-1-s))
                
                B[Tl[:,None],Tl] += 2*area[l]*length[m]/s*np.reshape(phi_edge@d, (3,3), order = 'F')


            #Intersection is vertex
            for m in vertex_edges:

                em = e[m]

                sh_nod = np.intersect1d(Tl,em)
                
                order = np.hstack((sh_nod,Tl[Tl != sh_nod]))

                B[order[:,None],order] += vertex_edgequad(Tl,em,sh_nod,p,s,normals[m],
                                        vpsi1_edge,vpsi2_edge,area[l],length[m],p_cube)
                
        else:
            
            idx_on_bndry = Tl[on_bndry]

            edgepatch1 = np.array(edgepatches[idx_on_bndry[0]])
            edgepatch2 = np.array(edgepatches[idx_on_bndry[1]])

            intersect_edge = np.union1d(edgepatch1,edgepatch2)
        
            edge_on_T = np.intersect1d(edgepatch1,edgepatch2)

            vertex_edges = np.setdiff1d(intersect_edge,edge_on_T)

            non_intersect_edge = np.setdiff1d(edge_number_array,intersect_edge)

            
            # Non intersecting edges
            for m in non_intersect_edge:

                pe = p[:,e[m]]
                ve = (pe[:,1]-pe[:,0])[None,:]*p_I_6[:,None] + pe[:,0]

                pos_diff, pos_diff_dist = pairwise_distances_with_vectors(vl,ve)

                d = np.multiply(normals[m]@pos_diff,pos_diff_dist**(-1-s))

                B[Tl[:,None],Tl] += 2*area[l]*length[m]/s*np.reshape(phi_edge@d, (3,3), order = 'F')

            # Intersection is vertex
            for m in vertex_edges:

                em = e[m]

                sh_nod = np.intersect1d(Tl,em)
                
                order = np.hstack((sh_nod,Tl[Tl != sh_nod]))

                B[order[:,None],order] += vertex_edgequad(Tl,em,sh_nod,p,s,normals[m],
                                        vpsi1_edge,vpsi2_edge,area[l],length[m],p_cube)
                
            #Edge lies on Tl

            m = edge_on_T[0]

            em = e[m]

            T_diff = np.setdiff1d(Tl,em)

            B[T_diff,T_diff] += edge_edgequad(Tl,em,T_diff,p,s,normals[m],
                                                epsi1_edge,epsi2_edge,epsi3_edge,
                                                area[l],length[m],p_I)


    return cns*B

def RegionalFractionalLaplaceStiffMat(p,t,s,
                                      N_p,N_T,area,
                                      bndrynode,patches):

    B = np.zeros((N_p,N_p))

    cns = 2**(2*s-1)*s*gamma(s+1)*rgamma(1-s)/np.pi

    vl = np.zeros((6,2))
    vm = np.zeros((6*N_T,2))
    ve = np.zeros((6,2))
    norms = np.zeros((36,N_T))

    ML = np.zeros((6,6,N_T))
    empty = np.zeros(N_T, dtype = np.int32)
    aux_ind = np.reshape(np.tile(np.arange(0, 3*N_T, 3), (6, 1)), -1)
    empty_vtx = np.zeros((2,3*N_T))
    BBm = np.zeros((2,2*N_T))

    for l in range(N_T):
        edge = np.concatenate((patches[t[l,0]],
                            patches[t[l,1]],
                            patches[t[l,2]]))
        
        nonempty, N = np.unique(edge, return_index=True)
        
        edge = np.delete(edge,N)

        vertex = np.setdiff1d(nonempty,edge)

        ll = N_T-l-np.sum(nonempty >= l)

        edge = edge[edge > l]
        vertex = vertex[vertex > l]

        empty[:ll] = np.setdiff1d(np.arange(l,N_T), nonempty)[:ll]

        empty_vtx[:,:3*ll] = p[:,t[empty[:ll],:].flatten()]
        
        Tl = t[l]

        xl = p[0,Tl]
        yl = p[1,Tl]

        K = np.array(((xl[1]-xl[0],xl[2]-xl[1]),(yl[1]-yl[0],yl[2]-yl[1])))
        
        #Identiacl elements
        B[Tl[:,None],Tl] += triangle_quad(K,s,tpsi1,tpsi2,tpsi3,area[l],p_I)
        
        #Disjoint elements
        BBm[0,:2*ll:2] = empty_vtx[0,1:3*ll:3]-empty_vtx[0,:3*ll:3]
        BBm[1,:2*ll:2] = empty_vtx[0,2:3*ll:3]-empty_vtx[0,1:3*ll:3]
        BBm[0,1:2*ll:2] = empty_vtx[1,1:3*ll:3]-empty_vtx[1,:3*ll:3]
        BBm[1,1:2*ll:2] = empty_vtx[1,2:3*ll:3]-empty_vtx[1,1:3*ll:3]
        
        vl = p_T_6 @ K.T + np.vstack([np.ones(6) * xl[0], np.ones(6) * yl[0]]).T
        for r in range(ll):
            vm[6*r:6*(r+1),:] = p_T_6@BBm[:,2*r:2*r+2] + empty_vtx[:,aux_ind[r]].T
        
        norms[:,:ll] = np.reshape(cdist(vl,vm[:6*ll, :]), (36,-1), order = 'F')**(-2-2*s)
        
        ML[:3,:3,:ll] = np.reshape(phiA@norms[:,:ll], (3,3,-1),order = 'F')
        ML[:3,3:6,:ll] = np.reshape(phiB@norms[:,:ll], (3,3,-1),order = 'F')
        ML[3:6,3:6,:ll] = np.reshape(phiD@norms[:,:ll], (3,3,-1),order = 'F')
        ML[3:6,:3,:ll] = np.transpose(ML[:3,3:6,:ll], (1,0,2))

        for m in range(ll):
            order = np.hstack((Tl,t[empty[m],:].flatten()))
            B[order[:,None],order] += 8*area[l]*area[empty[m]] * ML[:6,:6,m]


        #Vertex touching elements
        for m in vertex:
            Tm = t[m]
            T_com = np.intersect1d(Tl,Tm)
            order = np.hstack((T_com,Tl[Tl != T_com],Tm[Tm != T_com]))
            B[order[:,None],order] += 2*vertex_quad(Tl,Tm,T_com,p,s,vpsi1,vpsi2,
                                                    area[l],area[m],p_cube)

        #Edge touching elements
        for m in edge:
            Tm = t[m]
            T_diff = np.hstack([np.setdiff1d(Tl,Tm), np.setdiff1d(Tm,Tl)])
            order = np.hstack((Tl[Tl != T_diff[0]],T_diff))
            B[order[:,None],order] += 2*edge_quad(Tl,Tm,T_diff,p,s,epsi1,epsi2,epsi3,epsi4,epsi5,
                                            area[l],area[m],p_cube)
            
    return cns*B