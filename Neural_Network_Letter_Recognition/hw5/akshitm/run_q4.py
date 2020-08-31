import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    y_vals = [bbox[2]-bbox[0] for bbox in bboxes]
    avg_h = sum(y_vals)/len(y_vals) # average Height of boxes

    centers = [[(bbox[3] + bbox[1])//2, (bbox[2] + bbox[0])//2, bbox[3]-bbox[1],bbox[2]-bbox[0]] for bbox in bboxes]
    centers.sort(key = lambda i:i[1]) # Sort based on x1 values
    x_init = centers[0][1]

    row_val = []
    row_list= []

    for coord in centers:
        if coord[1] > x_init + avg_h:
            row_val = sorted(row_val, key=lambda coord: coord[0]) #sort based on y values
            row_list.append(row_val)
            x_init = coord[1]
            row_val = [coord]

        else:
            row_val.append(coord)


    row_val = sorted(row_val, key=lambda coord:coord[0]) #sort based on y values

    row_list.append(row_val) # coordinates of rectangle rows

    im_mat = []

    for row in row_list:
        im_row = []

        for x_cord, y_cord, w, h in row:
            im_crop = bw[y_cord - h//2 : y_cord + h//2, x_cord - w//2 : x_cord + w//2]  # crop the bounding boxes

    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
            # pad_h = (np.abs(h-w)//2) +(np.max([h,w])//20)
            if h > w:
                pad_h = h//20
                pad_w = (h-w)//2 + pad_h

            else:
                pad_w = w//20
                pad_h = (w-h)//2 + pad_h

            im_crop = np.pad(im_crop,((pad_h, pad_h),(pad_w,pad_w)),'constant',constant_values=(1,1))
            im_crop = skimage.transform.resize(im_crop, (32,32))
            im_crop = (skimage.morphology.erosion(im_crop)).T
            crop_flat = im_crop.flatten()
            im_row.append(crop_flat)

        im_arr = np.asarray(im_row)
        im_mat.append(im_arr)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    for im_arr in im_mat:
        h1 = forward(im_arr,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        pred_loc = np.argmax(probs, axis=1)
        pred_val = ''
        for i in pred_loc:
            pred_val +=  letters[i]
        print(pred_val)