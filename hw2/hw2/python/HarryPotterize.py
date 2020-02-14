import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac

#Write script for Q2.2.4
opts = get_opts()

#Read required images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover,cv_desk,opts)
bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)