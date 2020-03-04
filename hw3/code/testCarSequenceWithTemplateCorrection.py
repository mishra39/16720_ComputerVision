import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=2e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
frame_eval = [1, 100, 200, 300, 400]

rect_orig  = rect
epsilon = .01

frame_template = seq[:,:,0]  # Template frame
rect_all = []

frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0]
height = rect[3] - rect[1]
p0=np.zeros(2)
#Start the figure
fig,ax = plt.subplots(1)

for ind in range(frame_tot-1):
    frame_prev = seq[:,:,ind]
    frame1 = seq[:,:,ind+1]
    print(ind)

    p = LucasKanade(frame_prev ,frame1, rect,threshold,num_iters)
    # p0[0] = rect[0] + p[0] - rect_orig[0]
    # p0[1] = rect[1] + p[1] - rect_orig[1]
    p0 = np.array(p)
    p_star = LucasKanade(frame_template ,frame1, rect_orig,threshold, num_iters, p0)

    p_star_p0_norm = np.linalg.norm(p_star - p0)
    print(p_star,p0)
    if p_star_p0_norm < epsilon:
        rect_x = rect[0] + p[0]
        rect_y = rect[1] + p[1]
        rect[0] = rect[0] + p[0]
        rect[1] = rect[1] + p[1]
        rect[2] = rect[0] + width
        rect[3] = rect[1] + height
    else:
        rect_x = rect_orig[0] + p_star[0]
        rect_y = rect_orig[1] + p_star[1]
        rect[0] = rect_orig[0] + p_star[0]
        rect[1] = rect_orig[1] + p_star[1]
        rect[2] = rect[0] + width
        rect[3] = rect[1] + height
#
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