import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-4, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
frame_eval = [1, 20, 40, 60, 80]

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


    #Saving frames of interest
    if ind in frame_eval:
        rect_all.append(rect)
        plt.savefig('testGirlSequence_frame_' + str(ind) +'_threshold_'+ str(threshold) + 'iter_' + str(num_iters) +'.jpg')

    plt.show()
    plt.close()

np.save('girlseqrects.npy',np.asarray(rect_all))