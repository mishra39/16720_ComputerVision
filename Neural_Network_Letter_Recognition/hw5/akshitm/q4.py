import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None

    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    im_denoise = skimage.restoration.denoise_wavelet(image,multichannel=True) #denoise
    im_grey = skimage.color.rgb2gray(im_denoise) # grayscale
    im_thresh = skimage.filters.threshold_otsu(im_grey) # threshold
    bw = skimage.morphology.closing(im_grey < im_thresh, skimage.morphology.square(5))
    label_image = skimage.measure.label(bw,connectivity=2) # What does this do?
    regions = skimage.measure.regionprops(label_image)

    area_mean = 0
    area_mean = sum([i.area for i in regions])/len(regions)

    bboxes = [i.bbox for i in regions if i.area > area_mean/2]
    bw = (~bw).astype(np.float)
    return bboxes, bw