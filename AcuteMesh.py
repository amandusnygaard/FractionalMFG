import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class Mesh:

    def __init__(self, p, t,e, bndrynode,normals):
        
        '''
        p : 2xn_p array of points in 2D, where n_p is the number of points. 
            The array is such that p[:,n] gives the nth point
        t : N_Tx3 array of elements of mesh, where N_T is the number of elements.
            The array is such that t[l,:] gives the indices for the nodes in element T_l
        bndrynodes : boolian array that indicates if a node is a bounadry node.
        '''

        self.p = p.copy()
        self.t = t.copy()
        self.e = e.copy()
        self.bndrynode = bndrynode.copy()
        self.normals = normals.copy()

        self.N_p = p.shape[1]
        self.N_T = t.shape[0]
        self.N_e = e.shape[0]
        
        self.patches = None
        self.edgepatches = None

        self.h = None
        self.area = None
        self.length = None

        self.theta = 0

        return None
    
    def ComputeTheta(self):

        theta = 0

        for i in range(self.N_T):

            T = self.t[i,:]

            p0 = self.p[:,T[0]]
            p1 = self.p[:,T[1]]
            p2 = self.p[:,T[2]]

            v1 = p1-p0
            v2 = p2-p0
            v3 = p2-p1

            normv1 = np.linalg.norm(v1)
            normv2 = np.linalg.norm(v2)
            normv3 = np.linalg.norm(v3)

            costheta1 = np.inner(v1,v2)/(normv1*normv2)
            costheta2 = np.inner(v1,-v3)/(normv1*normv3)
            costheta3 = np.inner(v3,v2)/(normv3*normv2)

            theta_T = np.arccos(min(costheta1,costheta2,costheta3))

            theta = max(theta,theta_T)
        
        self.theta = theta
    
    def ComputeSizes(self, find_angle = False):

        self.h = np.zeros(self.N_T)
        self.area = np.zeros(self.N_T)
        
        if find_angle:
            theta = 0

        for i in range(self.N_T):

            T = self.t[i,:]

            p0 = self.p[:,T[0]]
            p1 = self.p[:,T[1]]
            p2 = self.p[:,T[2]]

            v1 = p1-p0
            v2 = p2-p0
            v3 = p2-p1

            normv1 = np.linalg.norm(v1)
            normv2 = np.linalg.norm(v2)
            normv3 = np.linalg.norm(v3)

            h_T = max(normv1,normv2,normv3)

            area_T = 0.5*abs(v1[0]*v2[1]-v1[1]*v2[0])

            if find_angle:
                costheta1 = np.inner(v1,v2)/(normv1*normv2)
                costheta2 = np.inner(v1,-v3)/(normv1*normv3)
                costheta3 = np.inner(v3,v2)/(normv3*normv2)

                theta_T = np.arccos(min(costheta1,costheta2,costheta3))
            
            self.h[i] = h_T
            self.area[i] = area_T

            if find_angle:
                theta = max(theta,theta_T)
        
        if find_angle:
            self.theta = theta

        self.length = np.zeros(self.N_e)

        for i in range(self.N_e):

            E = self.e[i]

            length_E = np.linalg.norm(self.p[:,E[0]]-self.p[:,E[1]])

            self.length[i] = length_E

        return None
            
    def FindPatches(self):
        '''
        Creates a list of neighboring nodes
        '''

        # deg = np.zeros(self.N_p, dtype = np.int32)

        # for i in range(self.N_T):
        #     deg[self.t[i,:]] += 1
        
        # self.patches = [np.zeros(degree) for degree in deg]
        self.patches = [[] for _ in range(self.N_p)]
        for i in range(self.N_T):
            self.patches[self.t[i,0]].append(i)
            self.patches[self.t[i,1]].append(i)
            self.patches[self.t[i,2]].append(i)

        self.edgepatches = [[] for _ in range(self.N_p)]
        for i in range(self.N_e):
            self.edgepatches[self.e[i,0]].append(i)
            self.edgepatches[self.e[i,1]].append(i)
        
        return None
    
    def FindInwardEdgeNormal(self,p1,p2,p3):
        edge_vec = p2-p1
        normal = np.array([-edge_vec[1], edge_vec[0]])  # 90° CCW
        normal /= np.linalg.norm(normal)
        inside_vec = p3-p1
        if np.dot(inside_vec, normal) < 0:  # ensure inward
            normal = -normal
        return normal
    
    def _Refine(self):

        new_p = []

        new_t = []

        new_e = []

        new_normals = []
        
        latest_index = self.N_p

        child_point_index = {}

        new_bndrynode = []

        for i in range(self.N_T):

            T = self.t[i]

            x0 = self.p[:,T[0]]
            x1 = self.p[:,T[1]]
            x2 = self.p[:,T[2]]
            
            x0onbndry = self.bndrynode[T[0]]
            x1onbndry = self.bndrynode[T[1]]
            x2onbndry = self.bndrynode[T[2]]

            x3 = 0.5*(x0+x1)
            x4 = 0.5*(x0+x2)
            x5 = 0.5*(x2+x1)

            x3onbndry = x0onbndry and x1onbndry
            x4onbndry = x0onbndry and x2onbndry
            x5onbndry = x1onbndry and x2onbndry

            idx3 = child_point_index.get(frozenset((T[0],T[1])))
            if idx3 == None:

                idx3 = latest_index
                child_point_index[frozenset((T[0],T[1]))] = idx3

                new_p.append(x3)

                latest_index += 1

                new_bndrynode.append(x3onbndry)


            idx4 = child_point_index.get(frozenset((T[0],T[2])))
            if idx4 == None:

                idx4 = latest_index
                child_point_index[frozenset((T[0],T[2]))] = idx4

                new_p.append(x4)

                latest_index += 1

                new_bndrynode.append(x4onbndry)

            idx5 = child_point_index.get(frozenset((T[2],T[1])))
            if idx5 == None:

                idx5 = latest_index
                child_point_index[frozenset((T[2],T[1]))] = idx5

                new_p.append(x5)

                latest_index += 1

                new_bndrynode.append(x5onbndry)

            new_t.extend([[T[0],idx3,idx4],
                        [T[1],idx3,idx5],
                        [T[2],idx4,idx5],
                        [idx3,idx4,idx5]])
            
            if x3onbndry:
                new_e.extend(((T[0],idx3),(idx3,T[1])))
                normal = self.FindInwardEdgeNormal(x0,x1,x2)
                new_normals.extend((normal,normal))

            if x4onbndry:
                new_e.extend(((T[0],idx4),(idx4,T[2])))
                normal = self.FindInwardEdgeNormal(x0,x2,x1)
                new_normals.extend((normal,normal))

            if x5onbndry:
                new_e.extend(((T[1],idx5),(idx5,T[2])))
                normal = self.FindInwardEdgeNormal(x1,x2,x0)
                new_normals.extend((normal,normal))
            

        self.p = np.hstack((self.p, np.array(new_p).T))
        self.t = np.array(new_t)
        self.e = np.array(new_e)

        self.bndrynode = np.append(self.bndrynode, np.array(new_bndrynode, dtype = np.bool))
        self.normals = np.array(new_normals)

        self.N_p = self.p.shape[1]
        self.N_T = self.t.shape[0]
        self.N_e = self.e.shape[0]

        return None
    
    def Refine(self,k = 1):

        for _ in range(k):
            self._Refine()

        return None
    
    def PlotMesh(self, highlight_bndrynodes = False, highlight_bndryedges = False,highlight_normals = False,
                 title = None,showgrid = False, path = None,showlegend = False):
        """
        Visualize the mesh, boundary nodes, edges, and normal vectors.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        
        # Plot triangulation
        triang = mtri.Triangulation(self.p[0, :], self.p[1,:], self.t)
        ax.triplot(triang, color='black', linewidth=1)

        # Plot boundary nodes
        if highlight_bndrynodes:
            ax.plot(self.p[0,self.bndrynode], self.p[1,self.bndrynode], 'ro', markersize=3, label="Boundary nodes")

        # Plot boundary edges and inward normal vectors

        if highlight_bndryedges or highlight_normals:
            for (i, j), normal in zip(self.e, self.normals):
                p1 = self.p[:,i]
                p2 = self.p[:,j]
                midpoint = (p1 + p2) / 2
                if highlight_bndryedges:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1.0)  # Edge
                if highlight_normals:
                    ax.arrow(midpoint[0], midpoint[1], 0.1 * normal[0], 0.1 * normal[1],
                            head_width=0.02, head_length=0.03, fc='blue', ec='blue')

        if not (title is None):
            ax.set_title(title)
        if showlegend:
            ax.legend(loc='upper right')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(showgrid)

        if not (path is None):
            plt.savefig(path)

        plt.show()

    def Copy(self):

        return Mesh(self.p,self.t,self.e,self.bndrynode,self.normals)
    

class Mesh1D:

    def __init__(self, p):

        self.p = p.copy()
        
        self.N_p = len(p)
        self.N_T = self.N_p-1

        self.h = p[1:]-p[:-1]

        return None
    
    def _Refine(self):

        p_ref = np.zeros(2*self.N_T+1)
        
        p_ref[::2] = self.p
        p_ref[1::2] = 0.5*(self.p[:-1]+self.p[1:])

        self.p = p_ref

        self.h = p_ref[1:]-p_ref[:-1]

        self.N_p = len(p_ref)
        self.N_T = self.N_p-1

        return None
    
    def Refine(self,k = 1):

        for _ in range(k):
            self._Refine()

        return None
    
    def __len__(self):
        return self.N_p
    
    def __getitem__(self, i):
        return self.p[i]