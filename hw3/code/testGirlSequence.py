import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
frame_eval = [1, 20, 40, 60, 80]

frame_template = seq[:,:,0]  # Template frame
rect_all = []

frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0]
height = rect[3] - rect[1]

#Start the figure
fig,ax = plt.subplots(1)

for ind in range(frame_tot):
    frame1 = seq[:,:,ind]
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

# np.save('testGirlSequence.npy',rect_all)