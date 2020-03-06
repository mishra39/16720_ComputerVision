import numpy as np
from scipy.interpolate import RectBivariateSpline
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import cv2
from numpy import matlib

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

    y_tl = rect[0]
    x_tl = rect[1]
    y_br = rect[2]
    x_br = rect[3]

    rect_y = int(y_br-y_tl+1)
    rect_x = int(x_br-x_tl+1)

    It_shape_0, It_shape_1 = It.shape

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


    # b_mat = np.zeros((rect_h*rect_w),1) # Matrix for change in pixel value difference between template and frame

    # initialize del_p ??????
    del_p = np.inf

    # Create a meshgrid
    mesh_h, mesh_w = np.meshgrid(np.arange(rect_y),np.arange(rect_x))
    max_iter = 0
    while del_p >= threshold and max_iter < num_iters:
        max_iter += 1
        # Warping: Translation
        x_tl_wrp, y_tl_wrp, x_br_wrp, y_br_wrp = x_tl + p[0], y_tl + p[1], x_br + p[0], y_br + p[1]

        # Meshgrid for translated rectangle
        x_wrp = np.linspace(x_tl_wrp, x_br_wrp, rect_x)
        y_wrp = np.linspace(y_tl_wrp, y_br_wrp, rect_y)
        mesh_xw, mesh_yw = np.meshgrid(x_wrp, y_wrp)

        # Inerpolation on the template
        It1_interp = It1_spline.ev(mesh_yw, mesh_xw)

        # Jacobian for warp and gradient for image

        It1_dx = It1_spline.ev(mesh_yw, mesh_xw,dx=1).flatten()
        It1_dy = It1_spline.ev(mesh_yw, mesh_xw,dy=1).flatten()

        b = (It_interp - It1_interp).reshape(-1,1)
        A = np.zeros(((rect_y*rect_x),2*rect_y*rect_x)) # Matrix for gradients

        for ind in range(rect_y*rect_x):
            A[ind,2*ind] = It1_dx[ind]
            A[ind,2*ind+1] = It1_dy[ind]


        jacb = matlib.repmat(np.eye(2), rect_y*rect_x,1)
        A_jac = np.dot(A,jacb)

        del_p = np.linalg.pinv(A_jac) @ b
        del_p = np.linalg.norm(del_p)

        # H = np.dot(np.transpose(A_mat),A_mat)  #Hessian Calculation
        # error = b
        # del_p1 = np.dot(np.transpose(A),b)
        # del_p = (np.dot(np.linalg.inv(H),del_p1)).flatten()
        p += del_p.T
        p = p.flatten()

    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
    parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold

    seq = np.load("../data/carseq.npy")
    rect = [59, 116, 145, 151]
    frame_eval = [1, 100, 200, 300, 400]

    frame_template = seq[:,:,0]  # Template frame
    rect_all = []

    frame_tot = seq.shape[2] # Total number of frames
    width = rect[2]-rect[0]
    height = rect[3] - rect[1]

    #Start the figure
    fig,ax = plt.subplots(1)

    for ind in range(frame_tot):
        frame_template = seq[:,:,ind]
        frame1 = seq[:,:,ind+1]
        print(ind)
        p = LucasKanade(frame_template ,frame1, rect,threshold,num_iters)
        rect_x = rect[0] + p[0]
        rect_y = rect[1] + p[1]
        img_patch = patches.Rectangle((rect_x,rect_y), width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')

        fig,ax = plt.subplots(1)
        ax.imshow(frame1,cmap='gray')
        ax.add_patch(img_patch)
        plt.axis('off')
        plt.show()
        # Saving frames of interest
        if ind in frame_eval:
            rect_save = [rect_x, rect_y, rect[2]+width, rect[3] + height]
            rect_all.append(rect_save)

# write your script here, we recommend the above libraries for making your animation
