import numpy as np

from AcuteMesh import Mesh

from FracLaplace import FractionalLaplaceStiffMat

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

mesh = Mesh(points,elements,edges,bndrynode,normals)
num_reff = 5
mesh.Refine(num_reff)
mesh.ComputeSizes()
mesh.FindPatches()

s = 0.75

fraclapl = FractionalLaplaceStiffMat(mesh.p,mesh.t,mesh.e,s,mesh.N_p,mesh.N_T,mesh.N_e,
                                     mesh.area,mesh.length,mesh.bndrynode,mesh.normals,
                                     mesh.patches,mesh.edgepatches)

np.savez('FracLapl_075_5ref.npz', fraclapl = fraclapl, s = s,p_ref = mesh.p)