import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


ratio = [0.7,0.7,0.7,0.1, 0.5,0.9]
sigma = [0.15,0.7,0.05,0.15,0.15,0.15]
test_arr = np.array([[0.15,0.7],[0.7,0.7],[0.05,0.7],[0.15,0.1],[0.15,0.5],[0.15,0.9]])

for row in range(test_arr.shape[0]):
    opts.ratio = test_arr[row][1]
    opts.sigma = test_arr[row][0]
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    
    #display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2)
    # for col in range(test_arr.shape[1]):