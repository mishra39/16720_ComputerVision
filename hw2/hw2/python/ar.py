import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt
import os, math, multiprocessing
import pdb
import imageio
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from loadVid import loadVid
from planarH import compositeH


def ar_func(ar_src, cv_cover, cv_cov_vid, opts):
    im = ar_src[44:311,:,:]
    cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))

    matches, locs1, locs2 = matchPics(np.transpose(cv_cover,(1,0,2)),np.transpose(cv_cov_vid,(1,0,2)),opts)
    bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)
    resize_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
    cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)),:]
    mask_one = np.ones(cropped_im.shape)

    warped_mask = cv2.warpPerspective(mask_one ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
    warped_im = cv2.warpPerspective(cropped_im ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
    non_zero_mask = np.nonzero(warped_mask)
    cv_cov_vid[non_zero_mask] = 0
    composite_img = warped_im + cv_cov_vid
    return composite_img


opts = get_opts()

opts.sigma = 0.2
opts.ratio = 0.9
opts.max_iters = 800

#Write script for Q3.1
ar_src = loadVid('../data/ar_source.mov')
cv_cover_im = cv2.imread('../data/cv_cover.jpg')
book = loadVid('../data/book.mov')

writer = imageio.get_writer('../results/test_parallel_2.avi', fps= 5)
n_cpu = multiprocessing.cpu_count()
tot_run = 50#ar_src.shape[0]
composite_img = []
cv_cover = [cv_cover_im for ind in range(tot_run)]
opts_func = [opts for ind in range(tot_run)]
args = list(zip(ar_src,cv_cover,book[0:ar_src.shape[0],:,:,:],opts_func))

pool_out = multiprocessing.Pool(n_cpu)
out_all = pool_out.starmap(ar_func,args)
out_all = np.asarray(out_all)
for ind in out_all:
    writer.append_data(ind)


writer.close()
# # Processing the video one frame at a time
# for frame_num in range():#ar_src.shape[0]):

#     print("Iteration: ", frame_num)
#     # Loading Frames
#     im = ar_src[frame_num,:,:,:]
#     im = im[45:310,:,:]
#     cv_cov_vid = book[frame_num,:,:,:]
#     cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
#     matches, locs1, locs2 = matchPics(np.transpose(cv_cover,(1,0,2)),np.transpose(cv_cov_vid,(1,0,2)),opts)
#     bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)
#     print(inliers)
#     # print(matches)
#     # Isotropic Cropping
#     resize_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))#cv2.resize(im, (cv_cover[1], im.shape[0]))  #im[int(im.shape[0]/2)-int(cv_cover.shape[0]/2) : int(im.shape[0]/2)+int(cv_cover.shape[0]/2) ,  int(im.shape[1]/2)-int(cv_cover.shape[1]/2) : int(im.shape[1]/2)+int(cv_cover.shape[1]/2) ,:]
#     cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)),:]
#     mask_one = np.ones(cropped_im.shape)
#     warped_mask = cv2.warpPerspective(mask_one ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))

#     warped_im = cv2.warpPerspective(cropped_im ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
#     non_zero_mask = np.nonzero(warped_mask)
#     cv_cov_vid[non_zero_mask] = 0
#     composite_img = warped_im + cv_cov_vid
#     # print("Warped size: ", warped_im.shape)
#     # print("Book frame size: ", cv_cov_vid.shape)
#     # print("Cv Cover size: ", cv_cover.shape)
#     # print("AR source size: ", cropped_im.shape)

#     # composite_img = compositeH(bestH2to1 ,cropped_im ,cv_cov_vid)
# writer.close()
# # plt.imshow(cropped_im)