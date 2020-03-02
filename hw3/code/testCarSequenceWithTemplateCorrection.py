import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
frame_eval = [1, 100, 200, 300, 400]

rect_init = rect
epsilon = 5
frame_template = seq[:,:,0]  # Template frame
rect_all = []

frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0]
height = rect[3] - rect[1]

#Start the figure
fig,ax = plt.subplots(1)


for ind in range(frame_tot-1):
    frame_prev = seq[:,:,ind]
    frame1 = seq[:,:,ind+1]
    print(ind)
    p = LucasKanade(frame_prev ,frame1, rect,threshold,num_iters)
    p0 = np.array([rect[1]-rect_init[1]+p[0],rect[0]-rect_init[0]+p[1]])

    p_star = LucasKanade(frame_template ,frame1, rect_init,threshold,num_iters,p0)
    p_star_p0_norm = np.linalg.norm(p_star - p0)

    frame_rect = seq[ind,:]
    frame_rect_height = frame_rect[3] - frame_rect[1]
    frame_rect_width = frame_rect[2] - frame_rect[0]


    if p_star_p0_norm < epsilon:
        rect_x, rect_y  = rect[0]+p[1], rect[1]+p[0] # Update rectangle

    else:
        rect = np.array([rect_init[0]+p_star[1],rect_init[1] + p_star[0],rect[0]+width, rect[1]+height])

    img_patch = patches.Rectangle((rect_x, rect_y),width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')
    # rect_all.append(rect)
    ax.add_patch(img_patch)
    plt.axis('off')

    img_patch2 = patches.Rectangle((frame_rect[0], frame_rect[1]),frame_rect_width, frame_rect_height,linewidth = 2,edgecolor = 'r', facecolor ='none')
    # rect_all.append(rect)
    ax.imshow(frame1,cmap='gray')
    ax.add_patch(img_patch)
    plt.axis('off')


    plt.show()
    # rect = rect[0]+p[1], rect[1]+p[0], rect[2]+p[1], rect[3]+p[0]

    # Saving frames of interest
    if ind in frame_eval:
        cv2.imwrite('../result/CarSequence_' + str(ind) + '.jpg',frame1)