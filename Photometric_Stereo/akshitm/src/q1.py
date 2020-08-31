# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2
from skimage import io
import skimage
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    x_ax = np.linspace(0,res[0]-1,res[0])
    y_ax = np.linspace(0,res[1]-1,res[1])
    [xx,yy] = np.meshgrid(x_ax,y_ax)

    radius = rad // pxSize
    normal = radius**2 - ((res[0]//2+(center[0]//pxSize)-xx)**2 + (res[1]//2 - center[1]//pxSize-yy)**2)
    mask = normal >=0
    normal = normal*mask

    idx = np.where(normal==0)
    normal[idx[0],idx[1]] = 1

    x_off = (xx - res[0]//2) / np.sqrt(normal)
    y_off = (yy - res[1]//2) / np.sqrt(normal)

    alb  = 0.5

    n = (alb * (light[0] * x_off - light[1] * y_off + light[2])) / np.sqrt(1+ x_off**2 + y_off**2)
    n_msk = n*mask
    idx2 = np.where(n_msk < 0)
    n_msk[idx2[0],idx2[1]] = 0

    image = n_msk *255 / np.max(n_msk)
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    im = skimage.io.imread("../data/input_" + str(1)+".tif")
    P = im.shape[0]*im.shape[1]
    I = np.empty([7,P])

    for n in range(7):
        im = skimage.io.imread("../data/input_" + str(n+1)+".tif")
        im_rgb = skimage.color.rgb2xyz(im)
        im_y = im_rgb[:,:,1]
        I[n,:] = im_y.flatten()

    L = np.load(path+"sources.npy")
    L = L.T
    s = im.shape[:2]
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    B = np.linalg.lstsq(L.T, I, rcond=None)[0]

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = []

    for i in range(B.shape[1]):
        rho = np.linalg.norm(B[:,i])
        B[:,i] = B[:,i]*(1/rho)
        albedos.append(rho)

    normals = B
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedoIm = np.reshape(albedos,s)
    normals += abs(np.min(normals))
    normals /= np.max(normals)

    d1 = normals[0,:].reshape(s)
    d2 = normals[1,:].reshape(s)
    d3 = normals[2,:].reshape(s)

    normalIm = np.dstack((d1,d2))
    normalIm = np.dstack((normalIm,d3))

    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    plt.imshow(normalIm, cmap='rainbow')
    plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    gx = normals[0,:]/-normals[2,:]
    gy = normals[1,:]/-normals[2,:]

    zx = np.reshape(gx,s)
    zy = np.reshape(gy,s)

    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i)

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    y = range(surface.shape[0])
    x = range(surface.shape[1])
    mesh_x, mesh_y = np.meshgrid(x,y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(mesh_x,mesh_y,surface,edgecolor = 'none',cmap=cm.coolwarm)

    ax.set_zlim3d(-100, 150)
    ax.set_ylim3d(0, 500)
    ax.set_xlim3d(0, 400)
    plt.show()

if __name__ == '__main__':

    # Put your main code here
    #1b
    light1 = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
    light2 = np.array([1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])
    light3 = np.array([-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])
    center = np.array([0,0,0])
    rad = 0.75
    res = np.array([3840,2160])
    pxSize = 0.0007

    im = renderNDotLSphere(center,rad,light1,pxSize,res)
    plt.imshow(im,cmap='gray')
    plt.show()
    im = renderNDotLSphere(center,rad,light2,pxSize,res)
    plt.imshow(im,cmap='gray')
    plt.show()
    im = renderNDotLSphere(center,rad,light3,pxSize,res)
    plt.imshow(im,cmap='gray')
    plt.show()


    I,L,s = loadData()
    B = estimatePseudonormalsCalibrated(I,L)
    albedos, normals = estimateAlbedosNormals(B)
    # albedoIm, normalIm  = displayAlbedosNormals(albedos,normals,s)

    # 1d
    u, sing, vh = np.linalg.svd(I,full_matrices=False)
    rank = u.shape[1]

    #1.f
    surface = estimateShape(normals, s)
    plotSurface(surface)