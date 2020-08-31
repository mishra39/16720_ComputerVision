import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy") # Load the video
rect = [59, 116, 145, 151] # Initial location of object
frame_eval = [1, 100, 200, 300, 400] # Frames to be saved
rect_orig  = rect
rect_all = []

frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0] + 1
height = rect[3] - rect[1] + 1


for ind in range(frame_tot-1):
    frame_template = seq[:,:,ind]
    frame1 = seq[:,:,ind+1]

    p = LucasKanade(frame_template ,frame1, rect,threshold,num_iters)

    rect[0] = rect[0] + p[0]
    rect[1] = rect[1] + p[1]
    rect[2] = rect[0] + width
    rect[3] = rect[1] + height
    img_patch = patches.Rectangle((rect[0],rect[1]), width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')

    fig,ax = plt.subplots(1)
    ax.imshow(frame1,cmap='gray')
    ax.add_patch(img_patch)
    plt.axis('off')


    # Saving frames of interest
    if ind in frame_eval:
        rect_all.append(rect)
        plt.savefig('testCarSequence_frame_' + str(ind) +'threshold_'+ str(threshold) + 'iter_' + str(num_iters) +'.jpg')

    plt.show()
    plt.close()
np.save('carseqrects.npy',np.asarray(rect_all))