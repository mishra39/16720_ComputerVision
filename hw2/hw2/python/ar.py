import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from loadVid import loadVid

opts = get_opts()
#Write script for Q3.1
ar_src = loadVid('../data/ar_source.mov')
book_vid = loadVid('../data/book.mov')
matches, locs1, locs2 = matchPics(ar_src,book_vid,opts)

#Processing the video one frame at a time
for frame_num in range(ar_src[:,:,frame_num]):
    frame = frame_num