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
import time

# def ar_func(ind,ar_src, cv_cover, cv_cov_vid, opts):
#     im = ar_src[44:311,:,:]
#     cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
#     # print('Frame Number: ',ind)
#     matches, locs1, locs2 = matchPics(np.transpose(cv_cover,(1,0,2)),np.transpose(cv_cov_vid,(1,0,2)),opts)
#     bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)
#     resize_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
#     cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)),:]
#     mask_one = np.ones(cropped_im.shape)

#     warped_mask = cv2.warpPerspective(mask_one ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
#     warped_im = cv2.warpPerspective(cropped_im ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
#     non_zero_mask = np.nonzero(warped_mask)
#     cv_cov_vid[non_zero_mask] = 0
#     composite_img = warped_im + cv_cov_vid
#     return composite_img

def ar_func(ind,ar_src, cv_cover, cv_cov_vid, opts):
    im = ar_src[44:311,:,:]
    cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
    print('Frame Number Function: ',ind)
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
frm_np = np.zeros(())
t = time.time()

ar_src = loadVid('../data/ar_source.mov')
cv_cover_im = cv2.imread('../data/cv_cover.jpg')
book = loadVid('../data/book.mov')
elapsed = time.time() - t
print("Elapsed time for loading videos: ", elapsed)



t2 = time.time()
# writer = imageio.get_writer('../results/forloop_final_ar.avi', fps= 25)


# # Processing the video one frame at a time
# for frame_num in range(ar_src.shape[0]):

#     print("Iteration: ", frame_num)
#     im = ar_src[44:311,:,:]
#     cropped_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
#     print('Frame Number: ',ind)
#     matches, locs1, locs2 = matchPics(cv_cover,cv_cov_vid,opts)
#     locs1 = locs1[matches[:,0],0:2]
#     locs2 = locs2[matches[:,1],0:2]
#     bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
#     resize_im =  cv2.resize(im,(im.shape[1],cv_cover.shape[0]))
#     cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover.shape[1]/2)),:]
#     composite_img = compositeH(bestH2to1, cropped_im, cv_cov_vid)
#     return composite_img
# writer.close()
# elapsed2 = time.time() - t2
# print("Total time for creating videos for loop: ", elapsed2)



t3 = time.time()
writer1 = imageio.get_writer('../results/norm_for_final.avi', fps= 25)
# writer2 = imageio.get_writer('../results/forloop2b_final_ar.avi', fps= 1)#25)

# Processing the video one frame at a time
for frame_num in range(ar_src.shape[0]):
    if (frame_num <50 or frame_num > 75):
        if (frame_num!=179 and frame_num!=253 and frame_num!=322 and frame_num!=364 and frame_num!=378 and frame_num!=384 and frame_num!=388 and frame_num!=393 and frame_num!=397 and frame_num!=398 and frame_num!=399 and frame_num!=422 and frame_num!=435 and frame_num!=436 and frame_num!=437 and frame_num!=438 ):
            print("Iteration: ", frame_num)
            composite_img = ar_func(frame_num, ar_src[frame_num],cv_cover_im,book[frame_num],opts)
            writer1.append_data(composite_img)
    # temp_b = composite_img[:,:,0]
    # temp_r = composite_img[:,:,2]
    # composite_img[:,:,0] = temp_r
    # composite_img[:,:,2] = temp_b
    # writer2.append_data(composite_img)

writer1.close()
# writer2.close()
elapsed3 = time.time() - t3
print("Total time for creating videos for loop: ", elapsed3)



# t4 = time.time()
# writer = imageio.get_writer('../results/parallel_final_ar.avi', fps= 25)
# n_cpu = multiprocessing.cpu_count()
# tot_run = ar_src.shape[0]
# composite_img = []
# idx = list(range(tot_run))
# cv_cover = [cv_cover_im for ind in range(tot_run)]
# opts_func = [opts for ind in range(tot_run)]
# args = list(zip(idx,ar_src,cv_cover,book[0:ar_src.shape[0],:,:,:], opts_func))

# pool_out = multiprocessing.Pool(n_cpu)
# out_all = pool_out.starmap(ar_func,args)
# out_all = np.asarray(out_all)
# for ind in out_all:
#     writer.append_data(ind)

# writer.close()
# elapsed4 = time.time() - t4
# print("Total time for creating videos parallel processing: ", elapsed4)







# im = ar_src[9,:,:,:]
# im = im[45:310,:,:]
# cv_cov_vid = book[9,:,:,:]
# cropped_im =  cv2.resize(im,(im.shape[1],cv_cover_im.shape[0]))
# matches, locs1, locs2 = matchPics(np.transpose(cv_cover_im,(1,0,2)),np.transpose(cv_cov_vid,(1,0,2)),opts)
# bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)

# # Isotropic Cropping
# resize_im =  cv2.resize(im,(im.shape[1],cv_cover_im.shape[0]))#cv2.resize(im, (cv_cover[1], im.shape[0]))  #im[int(im.shape[0]/2)-int(cv_cover.shape[0]/2) : int(im.shape[0]/2)+int(cv_cover.shape[0]/2) ,  int(im.shape[1]/2)-int(cv_cover.shape[1]/2) : int(im.shape[1]/2)+int(cv_cover.shape[1]/2) ,:]
# cropped_im = resize_im[:, int(cropped_im.shape[1]/2)-(int(cv_cover_im.shape[1]/2)) : int(cropped_im.shape[1]/2)+(int(cv_cover_im.shape[1]/2)),:]
# mask_one = np.ones(cropped_im.shape)
# warped_mask = cv2.warpPerspective(mask_one ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))

# warped_im = cv2.warpPerspective(cropped_im ,(bestH2to1) ,(cv_cov_vid.shape[1], cv_cov_vid.shape[0]))
# non_zero_mask = np.nonzero(warped_mask)
# cv_cov_vid[non_zero_mask] = 0
# composite_img = warped_im + cv_cov_vid
    # print("Warped size: ", warped_im.shape)
    # print("Book frame size: ", cv_cov_vid.shape)
    # print("Cv Cover size: ", cv_cover.shape)
    # print("AR source size: ", cropped_im.shape)









# # plt.imshow(cropped_im)