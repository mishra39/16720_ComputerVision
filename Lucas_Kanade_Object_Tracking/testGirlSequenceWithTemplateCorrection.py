import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=1, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]


frame_eval = [1, 20, 40, 60, 80]

rect_x = rect[0]
rect_y = rect[1]

rect_orig  = rect

frame_template = seq[:,:,0]  # Template frame
frame_prev = frame_template
rect_all = []

frame_tot = seq.shape[2] # Total number of frames
width = rect[2]-rect[0] + 1
height = rect[3] - rect[1] + 1
p0=np.zeros(2)
p_updt = np.zeros(2)
old_rect = np.load('carseqrects.npy')
old_num = 0
for ind in range(frame_tot-1):

    frame1 = seq[:,:,ind+1]

    p = LucasKanade(frame_prev ,frame1, rect,threshold,num_iters,p0)
    p_updt[0] = rect[0] + p[0] - rect_orig[0]
    p_updt[1] = rect[1] + p[1] - rect_orig[1]
    p0 = np.array(p)
    p_star = LucasKanade(frame_template ,frame1, rect_orig ,threshold, num_iters, p_updt)

    p_star_p0_norm = np.linalg.norm(p_star - p0)

    if p_star_p0_norm < template_threshold:
        tmp_var = p_star - [rect[0] - rect_orig[0], rect[1] -  rect_orig[1]]
        rect[0] = rect[0] + tmp_var[0]
        rect[1] = rect[1] + tmp_var[1]
        rect[2] = rect[2] + tmp_var[0]
        rect[3] = rect[3] + tmp_var[1]
        frame_prev = seq[:,:, ind+1]
        p0 = np.zeros(2)
    else:
        p0 = p

    rect_x = rect[0] + p[0]
    rect_y = rect[1] + p[1]
#
    img_patch = patches.Rectangle((rect_x,rect_y), width, height,linewidth = 2,edgecolor = 'r', facecolor ='none')

    fig,ax = plt.subplots(1)
    ax.imshow(frame1,cmap='gray')
    ax.add_patch(img_patch)
    plt.axis('off')


    # Saving frames of interest
    if ind in frame_eval:
        rect_x_old = old_rect[old_num,0]
        rect_y_old = old_rect[old_num,1]
        old_patch = patches.Rectangle((rect_x_old,rect_y_old), width, height,linewidth = 2,edgecolor = 'b', facecolor ='none')
        # ax.add_patch(old_patch)
        old_num += 1
        rect_all.append(rect)
        plt.savefig('GirlTemplateCorr_frame_' + str(ind) +'_threshold_'+ str(threshold) + 'iter_' + str(num_iters) +'.jpg')

    plt.show()
    plt.close()
np.save('girlseqrects-wcrt.npy',np.asarray(rect_all))