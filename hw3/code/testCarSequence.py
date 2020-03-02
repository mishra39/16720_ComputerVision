import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

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

frame_template = seq[:,:,0]  # Template frame
rect_all = []
rect_all.append(rect)
frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0]
height = rect[3] - rect[1]

#Start the figure
fig,ax = plt.subplots(1)


for ind in range(frame_tot):
    frame1 = seq[:,:,ind]
    frame2 = seq[:,:,ind+1]
    print(ind)
    p = LucasKanade(frame_template ,frame1, rect,threshold,num_iters)
    rect  = np.array([rect[0]+p[1], rect[1]+p[0], rect[2]+p[1],rect[3]+p[0]])  # Update rectangle
    img_patch = patches.Rectangle(rect,width, height,linewidth = 2,fill = False)
    rect_all.append(rect)
    ax.add_patch(img_patch)
    plt.imshow(frame2)
    plt.pause(0.02)
    ax.clear()

    # Saving frames of interest
    if ind in frame_eval:
        cv2.imwrite('../result/CarSequence_' + str(ind) + '.jpg',frame1)