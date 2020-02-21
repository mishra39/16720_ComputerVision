import numpy as np
import cv2
import pdb
import scipy
import skimage.io
import skimage.color
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt

from opts import get_opts
from matplotlib import pyplot as plt
from matchPics import matchPics
from statistics import stdev
from skimage import img_as_float64
opts = get_opts()

left = cv2.imread('../data/pano_left.jpg') #use PIL
right = cv2.imread('../data/pano_right.jpg')

matches, locs1, locs2 = matchPics(np.transpose(left,(1,0,2)),np.transpose(right,(1,0,2)),opts)



bestH2to1, inliers = computeH_ransac(matches,locs1, locs2, opts)

warped_im = cv2.warpPerspective(right,(bestH2to1) ,(np.max(im1.shape[1]), left.shape[0]))