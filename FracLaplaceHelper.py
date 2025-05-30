import numpy as np

p_I = np.array([0.5000,0.0820,0.9180,0.0159,0.9841,0.3379,0.6621,0.8067,0.1933])
w_I = np.array([0.1651,0.0903,0.0903,0.0406,0.0406,0.1562,0.1562,0.1303,0.1303])

p_T_6 = np.array([[0.5541, 0.4459],[0.5541, 0.1081],[0.8919, 0.4459],
                  [0.9084, 0.0916],[0.9084, 0.8168],[0.1832, 0.0916]])

p_cube = np.array([
[0.1127, 0.1127, 0.1127],
[0.1127, 0.1127, 0.5000],
[0.1127, 0.1127, 0.8873],
[0.1127, 0.5000, 0.1127],
[0.1127, 0.5000, 0.5000],
[0.1127, 0.5000, 0.8873],
[0.1127, 0.8873, 0.1127],
[0.1127, 0.8873, 0.5000],
[0.1127, 0.8873, 0.8873],
[0.5000, 0.1127, 0.1127],
[0.5000, 0.1127, 0.5000],
[0.5000, 0.1127, 0.8873],
[0.5000, 0.5000, 0.1127],
[0.5000, 0.5000, 0.5000],
[0.5000, 0.5000, 0.8873],
[0.5000, 0.8873, 0.1127],
[0.5000, 0.8873, 0.5000],
[0.5000, 0.8873, 0.8873],
[0.8873, 0.1127, 0.1127],
[0.8873, 0.1127, 0.5000],
[0.8873, 0.1127, 0.8873],
[0.8873, 0.5000, 0.1127],
[0.8873, 0.5000, 0.5000],
[0.8873, 0.5000, 0.8873],
[0.8873, 0.8873, 0.1127],
[0.8873, 0.8873, 0.5000],
[0.8873, 0.8873, 0.8873]])

w_cube =np.array([0.0214,0.0343,0.0214,0.0343,0.0549,0.0343,0.0214,0.0343,0.0214,
                  0.0343,0.0549,0.0343,0.0549,0.0878,0.0549,0.0343,0.0549,0.0343,
                  0.0214,0.0343,0.0214,0.0343,0.0549,0.0343,0.0214,0.0343,0.0214])


p_I_6 = np.array([
   0.03376524,
   0.16939531,
   0.38069041,
   0.61930959,
   0.83060469,
   0.96623476,
])

w_I_6 = np.array([
    0.08566225,
    0.18038079,
    0.23395697,
    0.23395697,
    0.18038079,
    0.08566225
])

def setdiff(A, B):
    A = np.array(A)  # Convert to NumPy array
    e = A.copy()  # Copy A to avoid modifying the original
    b = B - A[0] # Shift B values based on A[0]
    b = b[b >= 1]  # Remove values where b < 1
    b = b.astype(int)  # Convert to integer for indexing

    mask = np.ones(len(e), dtype=bool)  # Create a mask to track removals
    mask[b[b < len(e)]] = False  # Mark indices in b for removal
    return e[mask]  # Return filtered array

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def triangle_quad(Bl, s, psi1, psi2, psi3, areal, p_I):
    # Compute the coefficient
    coeff = (8 * areal**2) / ((4 - 2 * s) * (3 - 2 * s) * (2 - 2 * s))
    
    # Compute the transformed values
    term1 = np.sum((Bl @ np.vstack([p_I.T, np.ones(len(p_I))]))**2, axis=0) ** (-1 - s)
    term2 = np.sum((Bl @ np.vstack([np.ones(len(p_I)), p_I.T]))**2, axis=0) ** (-1 - s)
    term3 = np.sum((Bl @ np.vstack([p_I.T, p_I.T - np.ones(len(p_I))]))**2, axis=0) ** (-1 - s)


    # Compute ML matrix
    ML = coeff * np.reshape(
        psi1@term1.T + psi2@term2.T + psi3@term3.T,
        (3, 3)
    )
    
    return ML


def edge_quad(nodl, nodm, nod_diff, p, s, psi1, psi2, psi3, psi4, psi5, areal, aream, p_c):
    # Extract coordinates
    xm = p[0, nodm]
    ym = p[1, nodm]
    xl = p[0, nodl]
    yl = p[1, nodl]

    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]

    # Identify local and shared nodes
    local_l = np.where(nodl != nod_diff[0])[0]
    nsh_l = np.where(nodl == nod_diff[0])[0][0]
    nsh_m = np.where(nodm == nod_diff[1])[0][0]

    # Define P1 and P2
    P1 = np.array([xl[local_l[0]], yl[local_l[0]]])
    P2 = np.array([xl[local_l[1]], yl[local_l[1]]])

    # Compute transformation matrices Bl and Bm
    Bl = np.array([
        [P2[0] - P1[0], -P2[0] + xl[nsh_l]],
        [P2[1] - P1[1], -P2[1] + yl[nsh_l]]
    ])

    Bm = np.array([
        [P2[0] - P1[0], -P2[0] + xm[nsh_m]],
        [P2[1] - P1[1], -P2[1] + ym[nsh_m]]
    ])

    # Compute the reshaped matrix ML
    ML = (4 * areal * aream / (4 - 2 * s)) * np.reshape(
        psi1 @ np.sum((np.dot(np.column_stack((np.ones(len(x)), x * z)), Bl.T) - 
                       np.dot(np.column_stack((1 - x * y, x * (1 - y))), Bm.T))**2, axis=1)**(-1 - s) +
        psi2 @ np.sum((np.dot(np.column_stack((np.ones(len(x)), x)), Bl.T) - 
                       np.dot(np.column_stack((1 - x * y * z, x * y * (1 - z))), Bm.T))**2, axis=1)**(-1 - s) +
        psi3 @ np.sum((np.dot(np.column_stack(((1 - x * y), x * (1 - y))), Bl.T) - 
                       np.dot(np.column_stack((np.ones(len(x)), x * y * z)), Bm.T))**2, axis=1)**(-1 - s) +
        psi4 @ np.sum((np.dot(np.column_stack((1 - x * y * z, x * y * (1 - z))), Bl.T) - 
                       np.dot(np.column_stack((np.ones(len(x)), x)), Bm.T))**2, axis=1)**(-1 - s) +
        psi5 @ np.sum((np.dot(np.column_stack((1 - x * y * z, x * (1 - y * z))), Bl.T) - 
                       np.dot(np.column_stack((np.ones(len(x)), x * y)), Bm.T))**2, axis=1)**(-1 - s),
        (4, 4)
    )

    return ML

def vertex_quad(nodl, nodm, sh_nod, p, s, psi1, psi2, areal, aream, p_c):
    # Extract coordinates
    xm = p[0, nodm]
    ym = p[1, nodm]
    xl = p[0, nodl]
    yl = p[1, nodl]
    
    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]
    
    # Find local and non-shared nodes
    local_l = np.where(nodl == sh_nod)[0]
    nsh_l = np.where(nodl != sh_nod)[0]
    nsh_m = np.where(nodm != sh_nod)[0]

    
    p_c = np.array([xl[local_l[0]], yl[local_l[0]]])
    
    # Compute transformation matrices Bl and Bm
    Bl = np.array([
        [xl[nsh_l[0]] - p_c[0], xl[nsh_l[1]] - xl[nsh_l[0]]],
        [yl[nsh_l[0]] - p_c[1], yl[nsh_l[1]] - yl[nsh_l[0]]]
    ])
    
    Bm = np.array([
        [xm[nsh_m[0]] - p_c[0], xm[nsh_m[1]] - xm[nsh_m[0]]],
        [ym[nsh_m[0]] - p_c[1], ym[nsh_m[1]] - ym[nsh_m[0]]]
    ])
    # Compute ML
    ML = (4 * areal * aream / (4 - 2 * s)) * np.reshape(
        psi1 @ np.sum(
            (np.dot(np.column_stack((np.ones(len(x)), x)), Bl.T) -
             np.dot(np.column_stack((y, y * z)), Bm.T))**2, axis=1
        )**(-1 - s) +
        psi2 @ np.sum(
            (np.dot(np.column_stack((np.ones(len(x)), x)), Bm.T) -
             np.dot(np.column_stack((y, y * z)), Bl.T))**2, axis=1
        )**(-1 - s),
        (5, 5)
    )
    
    return ML

def pairwise_distances_with_vectors(vl, ve):
    nl = vl.shape[0]
    ne = ve.shape[0]
    distsquare = np.zeros(nl*ne)
    pairs = np.zeros((2,nl*ne))
    for i in range(ne):
        for j in range(nl):
            distsquare[i*ne + j] = (ve[i,0]-vl[j,0])**2 + (ve[i,1]-vl[j,1])**2
            pairs[:,i*ne+j] = vl[j]-ve[i]
    return pairs,distsquare


def vertex_edgequad(nodl,nodm,sh_nod,p,s,normalm,psi1,psi2,areal,lengthm,p_quad):
    xm = p[0, nodm]
    ym = p[1, nodm]
    xl = p[0, nodl]
    yl = p[1, nodl]

    x = p_quad[:, 0]
    y = p_quad[:, 1]
    z = p_quad[:, 2]

    # Find local and non-shared nodes
    local_l = np.where(nodl == sh_nod)[0]
    nsh_l = np.where(nodl != sh_nod)[0]
    nsh_m = np.where(nodm != sh_nod)[0]

    p_sh = np.array([xl[local_l[0]],yl[local_l[0]]])

    # Compute transformation matrices Bl and Bm
    Bl = np.array([
        [xl[nsh_l[0]] - p_sh[0], xl[nsh_l[1]] - xl[nsh_l[0]]],
        [yl[nsh_l[0]] - p_sh[1], yl[nsh_l[1]] - yl[nsh_l[0]]]
    ])
    
    Bm = np.array([xm[nsh_m[0]] - p_sh[0],ym[nsh_m[0]] - p_sh[1]])

    #Compute ML

    Js = x**(1-2*s)
    
    p1 = np.dot(np.column_stack([np.ones(len(y)),y]),Bl.T)-Bm[None,:]*z[:,None]

    p2 = np.dot(np.column_stack([y,y*z]),Bl.T)-Bm[None,:]*np.ones(len(x))[:,None]

    ML = (2*areal*lengthm/s)*np.reshape(
        psi1@(Js*np.dot(p1,normalm)*np.sum((p1**2),axis = 1)**(-1-s)) +
        psi2@(Js*np.dot(p2,normalm)*np.sum((p2**2),axis = 1)**(-1-s)),
    shape = (3,3))
    
    return ML

def edge_edgequad(nodl,nodm,nod_diff,p,s,normalm,psi1,psi2,psi3,areal,lengthm,p_quad):
    # Extract coordinates
    xm = p[0, nodm]
    ym = p[1, nodm]
    xl = p[0, nodl]
    yl = p[1, nodl]

    # Identify local and shared nodes
    local_l = np.where(nodl != nod_diff)[0]
    nsh = np.where(nodl == nod_diff)[0]

    Bl = np.array([[xl[local_l[1]]-xl[local_l[0]],xl[nsh[0]]-xl[local_l[0]]],
                   [yl[local_l[1]]-yl[local_l[0]],yl[nsh[0]]-yl[local_l[0]]]])
    
    p1 = np.dot(np.column_stack((1-p_quad,p_quad)),Bl.T)
    p2 = np.dot(np.column_stack((p_quad-1,np.ones(len(p_quad)))),Bl.T)
    p3 = np.dot(np.column_stack((-np.ones(len(p_quad)),p_quad)),Bl.T)
    
    ML = (2*areal*lengthm / (s*(3-2*s)*(4-2*s)))*(
        np.dot(psi1,np.dot(p1,normalm)*np.sum((p1**2),axis = 1)**(-1-s)) +
        np.dot(psi2,np.dot(p2,normalm)*np.sum((p2**2),axis = 1)**(-1-s)) +
        np.dot(psi3,np.dot(p3,normalm)*np.sum((p3**2),axis = 1)**(-1-s)) 
    )

    return ML
