import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import argparse
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)

    del_p = np.inf

    # Compute Image gradient
    I_dy, I_dx = np.gradient(It1)

    num_iter = 0


    while del_p >= threshold and num_iter <= num_iters:

        num_iter = num_iter + 1
        M_copy = np.copy(M)
        M_copy[0:2, 0:2] = np.fliplr(M_copy[0:2, 0:2])
        M_copy = np.flipud(M_copy)
        print(num_iter)

        M_copy = np.append(M_copy, np.array([[0,0,1]]), axis = 0)
        # Applying affine warp
        It1_warp = affine_transform(It1, M_copy, cval = -7)
        Idx_warp = affine_transform(I_dx, M_copy, cval = -7)
        Idy_warp = affine_transform(I_dy, M_copy, cval = -7)
        warp_points = np.where(It1_warp !=-7)

        b = It[warp_points] - It1_warp[warp_points]

        A = np.zeros([b.size,6])

        A[:,0] = I_dx[warp_points] * warp_points[1]
        A[:,1] = I_dx[warp_points] * warp_points[0]
        A[:,2] = Idx_warp[warp_points]
        A[:,3] = I_dy[warp_points] * warp_points[1]
        A[:,4] = I_dy[warp_points] * warp_points[0]
        A[:,5] = Idy_warp[warp_points]

        del_p = np.linalg.lstsq(A,b, rcond = -1)
        del_p = del_p[0]
        del_p_norm = np.linalg.norm(del_p) < threshold

        if del_p_norm < threshold:
            break;
        else:
            M += del_p.reshape(M.shape)
    return M


if __name__ == "__main__":
    # write your script here, we recommend the above libraries for making your animation
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
    parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
    args = parser.parse_args()
    num_iters = args.num_iters
    threshold = args.threshold

    seq = np.load("../data/carseq.npy")
    rect = [59, 116, 145, 151]
    frame_eval = [1, 100, 200, 300, 400]
    rect_orig  = rect
    frame_template = seq[:,:,0]  # Template frame
    rect_all = []

    frame_tot = seq.shape[2] # Total number of frames
    width = rect[2]-rect[0]
    height = rect[3] - rect[1]

    #Start the figure
    fig,ax = plt.subplots(1)

    for ind in range(frame_tot-1):
        frame_template = seq[:,:,ind]
        frame1 = seq[:,:,ind+1]
        print(ind)
        p = LucasKanadeAffine(frame_template ,frame1, threshold,num_iters)
        # rect_x = rect[0] + p[0]
        # rect_y = rect[1] + p[1]
        # rect[0] = rect[0] + p[0]
        # rect[1] = rect[1] + p[1]
        # rect[2] = rect[0] + width
        # rect[3] = rect[1] + height
        # img_patch = patches.Rectangle((rect_x,rect_y), width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')

        # fig,ax = plt.subplots(1)
        # ax.imshow(frame1,cmap='gray')
        # ax.add_patch(img_patch)
        # plt.axis('off')
        # plt.show()
        # # Saving frames of interest
        # if ind in frame_eval:
        #     rect_save = [rect_x, rect_y, rect[2]+width, rect[3] + height]
        #     rect_all.append(rect_save)