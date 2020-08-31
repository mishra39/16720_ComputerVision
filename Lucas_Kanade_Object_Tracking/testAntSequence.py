import argparse
import numpy as np
import matplotlib.pyplot as plt
from SubtractDominantMotion import SubtractDominantMotion
from scipy import ndimage


# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.05, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
frame_eval = [30, 60, 90, 120]

frame_tot = seq.shape[2] # Total number of frames
iter_num = 0

for ind in range(frame_tot-1):

    frame_prev = seq[:,:,ind]
    frame_curr = seq[:,:,ind+1]
    mask = SubtractDominantMotion(frame_prev, frame_curr, threshold, num_iters, tolerance)

    mask = ndimage.morphology.binary_erosion(mask, structure = np.eye(3))
    mask = ndimage.morphology.binary_dilation(mask, structure = np.ones((5,5)))

    ant = np.where(mask == 1)

    fig = plt.figure()
    plt.imshow(frame_prev, cmap = 'gray')
    fig, = plt.plot(ant[1], ant[0], '.',color='b')
    plt.axis('off')

    if ind in frame_eval:
        plt.savefig('testAntSequence_Inverse_frame_' + str(ind) +'_threshold_'+ str(threshold) + '_iter_' + str(num_iters) +'.jpg')

    plt.show()