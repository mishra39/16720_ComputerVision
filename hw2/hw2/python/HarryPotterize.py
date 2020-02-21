import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import time
import pdb
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

#Read required images
cv_cover = cv2.imread('../data/cv_cover.jpg') #use PIL
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
# matches, locs1, locs2 = matchPics(np.transpose(cv_cover,(1,0,2)),np.transpose(cv_desk,(1,0,2)),opts)
t = time.time()
matches, locs1, locs2 = matchPics(cv_cover,cv_desk,opts)
elapsed = t-time.time()

# max_iters = [100, 500, 1000, 5000, 10000, 5000, 5000, 5000, 5000]
# tol = [2.0,2.0,2.0,2.0,2.0, 1.0, 5.0, 7.0, 10.0]
hp_cover_resize = cv2.resize(hp_cover,(cv_cover.shape[1], cv_cover.shape[0]))

# for ind in range(len(max_iters)):
# opts.max_iters = max_iters[ind]
# opts.inlier_tol = tol[ind]
locs1 = locs1[matches[:,0],0:2]
locs2 = locs2[matches[:,1],0:2]
bestH2to1, inliers = computeH_ransac(locs1, locs2, opts)
composite_img = compositeH(bestH2to1 ,hp_cover_resize,cv_desk)
cv2.imwrite('../results/no_match_harry_poterize_debug.jpg',composite_img)
