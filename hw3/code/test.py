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


    It_shape_0, It_shape_1 = It.shape
    It1_shape_0, It1_shape_1 = It1.shape

    # Interpolate the image
    It_spline = RectBivariateSpline(np.arange(It_shape_0), np.arange(It_shape_1),It)
    It1_spline = RectBivariateSpline(np.arange(It1_shape_0), np.arange(It1_shape_1),It1)

    # Rectangle Coordinates
    x_tl = rect[0]
    y_tl = rect[1]
    x_br = rect[2]
    y_br = rect[3]

    # rectanglular window dimensions
    rect_y = int((y_br-y_tl+1))
    rect_x = int((x_br-x_tl+1))

    # initialize delta p
    del_p = np.inf

    # Create a meshgrid with rectangle window coordinates
    x_width = np.linspace(x_tl, x_br, rect_x)
    y_height = np.linspace(y_tl, y_br, rect_y)
    mesh_x, mesh_y = np.meshgrid(x_width,y_height)

    # Initial inerpolation on the template
    It_interp = It_spline.ev(mesh_y,mesh_x)

    # Compute Image gradient
    I_dy, I_dx = np.gradient(It1)

    # pdb.set_trace()
    num_iter = 0
    while del_p >= threshold and num_iter <= num_iters:

        num_iter = num_iter + 1

        # Warping with Translation
        x_tl_wrp, y_tl_wrp, x_br_wrp, y_br_wrp = x_tl + p[0], y_tl + p[1], x_br + p[0], y_br + p[1]

        # Meshgrid for translated rectangle
        x_wrp = np.linspace(x_tl_wrp, x_br_wrp, rect_x)
        y_wrp = np.linspace(y_tl_wrp, y_br_wrp, rect_y)
        mesh_xw, mesh_yw = np.meshgrid(x_wrp, y_wrp)



        # gradient for image
        dx_spline = RectBivariateSpline(np.arange(It1_shape_0), np.arange(It1_shape_1), I_dx)
        dy_spline = RectBivariateSpline(np.arange(It1_shape_0), np.arange(It1_shape_1), I_dy)

        # Interpolate second Image
        It1_interp = It1_spline.ev(mesh_yw, mesh_xw)
        It1_interp_x = dx_spline.ev(mesh_yw, mesh_xw)
        It1_interp_y = dy_spline.ev(mesh_yw, mesh_xw)
        # A = np.vstack((It1_interp_x.flatten(), It1_interp_y.flatten())
        b = (It_interp - It1_interp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=10, help='number of iterations of Lucas-Kanade')
    parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold

    seq = np.load("../data/carseq.npy")
    rect = [59, 116, 145, 151]
    frame_eval = [1, 100, 200, 300, 400]

    frame_template = seq[:,:,0]  # Template frame
    rect_all = []
    rect_all.append(rect)
    frame_tot = seq.shape[2] # Total number of frames
    width = rect[2]-rect[0]
    height = rect[3] - rect[1]

    #Start the figure
    fig,ax = plt.subplots(1)


    for ind in range(frame_tot):
        frame_prev = seq[:,:,ind]
        frame1 = seq[:,:,ind]
        print(ind)
        p = LucasKanade(frame_template ,frame1, rect,threshold,num_iters)
        rect_x, rect_y  = rect[0]+p[1], rect[1]+p[0] # Update rectangle
        # rect = rect[0]+p[1], rect[1]+p[0], rect[2]+p[1], rect[3]+p[0]
        img_patch = patches.Rectangle((rect_x, rect_y),width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')
        # rect_all.append(rect)
        fig,ax = plt.subplots(1)
        ax.imshow(frame1,cmap='gray')
        ax.add_patch(img_patch)
        plt.axis('off')
        plt.show()
        # Saving frames of interest
        if ind in frame_eval:
            cv2.imwrite('../result/CarSequence_' + str(ind) + '.jpg',frame1)