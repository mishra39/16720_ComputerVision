import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

#Read required images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(np.transpose(cv_cover,(1,0,2)),np.transpose(cv_desk,(1,0,2)),opts)
bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)
hp_cover_resize = cv2.resize(hp_cover,(cv_cover.shape[1], cv_cover.shape[0]))
warped_im = cv2.warpPerspective(hp_cover_resize,(bestH2to1) ,(cv_desk.shape[1], cv_desk.shape[0]))
plt.imshow(warped_im)
composite_img = compositeH(bestH2to1 ,hp_cover_resize,cv_desk)
