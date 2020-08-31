import numpy as np
import cv2
from matchPics import matchPics
import scipy
from opts import get_opts
from matplotlib import pyplot as plt
from helper import plotMatches

#Q2.1.6
num_match = []
rotn_mat = []
opts = get_opts()
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')

rotn = 0

for i in range(36):

   	#Rotate Image
    print('Image rotated by ',rotn,' degrees')
    cv_rot = scipy.ndimage.rotate(cv_cover,rotn)
    rotn_mat.append(rotn)
    rotn = rotn + 10

   	#Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, cv_rot, opts)
    plotMatches(cv_cover, cv_rot, matches, locs1, locs2)
    match_count  = matches.shape[0]
    num_match.append(match_count)
   	#Update histogram

#Display histogram

print(num_match)
print(rotn_mat)
angles = np.transpose(np.array(rotn_mat))

y_pos = np.arange(len(angles))
plt.bar(y_pos,np.transpose(np.array(num_match)), align='center', alpha=0.5)
plt.xticks(y_pos, angles)
plt.ylabel('Number of matches')
plt.title('Number of Matches as the image is rotated')
plt.show()