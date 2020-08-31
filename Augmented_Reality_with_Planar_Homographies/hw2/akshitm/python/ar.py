import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt

import pdb
import imageio
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from loadVid import loadVid
from planarH import compositeH



def ar_func(ind,ar_src, cv_cover, cv_cov_vid, opts):
    im = ar_src[44:311,:,:]
    cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
    matches, locs1, locs2 = matchPics(cv_cover,cv_cov_vid,opts)
    locs1 = locs1[matches[:,0],0:2]
    locs2 = locs2[matches[:,1],0:2]
    bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
    resize_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
    cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)),:]
    composite_img = compositeH(bestH2to1, cropped_im, cv_cov_vid)
    return composite_img

opts = get_opts()

#Write script for Q3.1

ar_src = loadVid('../data/ar_source.mov')
cv_cover_im = cv2.imread('../data/cv_cover.jpg')
book = loadVid('../data/book.mov')
vid_out = cv2.VideoWriter('../result/ar.avi',cv2.VideoWriter_fourcc('X','V','I','D'),25,(book[1].shape[1],book[0].shape[0]))

# Processing the video one frame at a time
for frame_num in range(ar_src.shape[0]):
    if (frame_num <50 or frame_num > 75):
        if (frame_num!= 81 and frame_num!=119 and frame_num!=179 and frame_num!=196 and frame_num!=253 and frame_num!=322 and frame_num!=364 and frame_num!=378 and frame_num!=384 and frame_num!=388 and frame_num!=393 and frame_num!=397 and frame_num!=398 and frame_num!=399 and frame_num!=422 and frame_num!=435 and frame_num!=436 and frame_num!=437 and frame_num!=438 and frame_num!=447):
            composite_img = ar_func(frame_num, ar_src[frame_num],cv_cover_im,book[frame_num],opts)
            vid_out.write(composite_img)

cv2.destroyAllWindows()
vid_out.release()