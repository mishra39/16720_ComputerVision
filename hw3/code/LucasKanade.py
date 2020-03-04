import numpy as np
from scipy.interpolate import RectBivariateSpline
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0

    # Rectangle Coordinates
    x_tl = rect[0]
    y_tl = rect[1]
    x_br = rect[2]
    y_br = rect[3]
    It_shape_0, It_shape_1 = It.shape

    # rectanglular window dimensions
    rect_y = int((y_br-y_tl+1))
    rect_x = int((x_br-x_tl+1))

    # initialize delta p
    del_p = np.inf

    # Spline for the template image
    It_spline = RectBivariateSpline(np.arange(It_shape_0),np.arange(It_shape_1),It)
    # Interpolate the image Splines for images
    It1_spline = RectBivariateSpline(np.arange(It_shape_0),np.arange(It_shape_1),It1)

    # Create a meshgrid with rectangle window coordinates
    x_width = np.linspace(x_tl, x_br, rect_x)
    y_height = np.linspace(y_tl, y_br, rect_y)
    mesh_x, mesh_y = np.meshgrid(x_width,y_height)
    It_interp = It_spline.ev(mesh_y, mesh_x) # Applying the spline to image 1

    # Compute Image gradient
    I_dy, I_dx = np.gradient(It1)

    # gradient for image
    dx_spline = RectBivariateSpline(np.arange(It_shape_0),np.arange(It_shape_1), I_dx)
    dy_spline = RectBivariateSpline(np.arange(It_shape_0),np.arange(It_shape_1), I_dy)

    num_iter = 0

    while del_p >= threshold and num_iter <= num_iters:

        num_iter = num_iter + 1
        # Warping with Translation
        x_tl_wrp, y_tl_wrp, x_br_wrp, y_br_wrp = x_tl + p[0], y_tl + p[1], x_br + p[0], y_br + p[1]

        # Meshgrid for translated rectangle
        x_wrp = np.linspace(x_tl_wrp, x_br_wrp, rect_x)
        y_wrp = np.linspace(y_tl_wrp, y_br_wrp, rect_y)
        mesh_xw, mesh_yw = np.meshgrid(x_wrp, y_wrp)

        # Inerpolation on the template
        It1_interp = It1_spline.ev(mesh_yw, mesh_xw)
        It1_interp_x = dx_spline.ev(mesh_yw, mesh_xw)
        It1_interp_y = dy_spline.ev(mesh_yw, mesh_xw)

        A = np.vstack((It1_interp_x.flatten(), It1_interp_y.flatten())).T

        b = (It_interp - It1_interp).reshape(-1,1)  # Error calculation

        H = np.dot(np.transpose(A),A)
        H_inv = np.linalg.inv(H)
        A_t_b = np.dot(np.transpose(A),b)
        del_p = np.dot(H_inv,A_t_b)
        p[0] += del_p[0]
        p[1] += del_p[1]
        del_p = np.linalg.norm(del_p)

    return p