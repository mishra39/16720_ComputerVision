import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    lab_img = skimage.color.rgb2lab(img) #Convert image into the lab color space 
    filter_scales = opts.filter_scales
    scale_size = len(filter_scales) #Number of filter scales
    filt_num = 4 #Number of filters
    filt_bank = scale_size*filt_num
    
    #Filters to use
    scipy.ndimage.gaussian_filter()
    scipy.ndimage.gaussian_laplace()
    #how to calculate derivatice of gaussian in x and y?
    # ----- TODO -----
    H,W = img.shape[0],img.shape[1]
    filter_responses = np.zeros((H,W,3*filt_bank))
    
    for loc, in range(0,):
        
        gauss_img = scipy.ndimage.gaussian_filter(lab_img[:,:,channel],filter_scales[loc]) #apply gaussian filter to the lab image
        laplace_img = scipy.ndimage.gaussian_laplace(lab_img[:,:,channel],filter_scales[loc]) #apply gaussian laplace filter to the lab image
        
    
    
    
    filtimg = scipy.ndimage.gaussian_filter(img,sigma = 5)
    
    return filtimg

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files_small.txt')).read().splitlines()
    # ----- TODO -----
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    pass

