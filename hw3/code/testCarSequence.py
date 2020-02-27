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
rect_all = []
rect_all.append(rect)
frame_tot = seq.shape[2]
frame_template = seq[:,:,0]  # Template frame

for ind in range(1,frame_tot):
    It1_frame = seq[:,:,ind]
    p_out = LucasKanade(frame_template,It1_frame,rect,threshold,num_iters)

    rect_next = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0],rect[1]+p[1]]  # Update rectangle

for ind in frame_eval:
    frame = seq[:,:,ind]
