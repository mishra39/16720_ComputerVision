import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# I1, I2 : Images to match
# opts: input opts

def matchPics(I1, I2, opts):


    # Convert Images to GrayScale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)


    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'
    # print('Sigma is: ',sigma, 'and Ratio is: ', ratio)


    # Detect Features in Both Images
    locs1 = corner_detection(I1, sigma) #locations of corners
    locs2 = corner_detection(I2, sigma) #locations of corners using FAST

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2