import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2, opts):
    
    
    # Convert Images to GrayScale
    I1 = np.dot(I1, [0.299, 0.587, 0.114])
    I2 = np.dot(I2, [0.299, 0.587, 0.114])
    
    
    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'
    print(sigma,ratio)
    

    # Detect Features in Both Images
    locs1 = corner_detection(I1, sigma)
    locs2 = corner_detection(I2, sigma)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)
            
        # I1, I2 : Images to match
        # opts: input opts

    

    return matches, locs1, locs2