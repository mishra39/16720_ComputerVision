# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface
from utils import enforceIntegrability
from mpl_toolkits.mplot3d import Axes3D

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u,sing,vh = np.linalg.svd(I,full_matrices=False)
    sing[3:] = 0
    sing_r3 = np.diag(sing[:3])
    vh_r3 = vh[:3,:]
    u_r3 = u[:,:3]
    B = np.dot(np.sqrt(sing_r3),vh_r3)
    L = np.dot(u_r3,np.sqrt(sing_r3))
    L = L.T

    return B, L


if __name__ == "__main__":

    # Put your main code here
    #2 c
    I,L0,s = loadData()
    B,L = estimatePseudonormalsUncalibrated(I)

    #2.d
    # albedos, normals = estimateAlbedosNormals(B)
    # surface = estimateShape(normals,s)
    # plotSurface(surface)

    # 2.e
    intg = enforceIntegrability(B,s)
    albedos, normals = estimateAlbedosNormals(intg)
    # surface = estimateShape(normals,s)
    # plotSurface(surface)

    # 2.f
    mu = 0
    v = 0
    lam = 100
    G = np.array([[1,0,0],[0,1,0],[mu,v,lam]])
    G_T_B = np.dot(np.linalg.inv(G.T),B)

    intg_f = enforceIntegrability(G_T_B,s)
    albedos, normals = estimateAlbedosNormals(intg_f)
    surface = estimateShape(normals,s)
    plotSurface(surface)
