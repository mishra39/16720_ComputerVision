import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from numpy import matlib
from multiprocessing import Pool
from opts import get_opts
import sklearn.cluster

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    #Check the number of input channels
    img_dim = len(img.shape)

    if img_dim < 3:
        img = np.matlib.repmat(img,3,1)
        
    if img.shape[-1] > 3: #What to do??
        img = img[:,:,0:3]
    
    lab_img = skimage.color.rgb2lab(img) #Convert image into the lab color space 
    filter_scales = opts.filter_scales

    scale_size = len(filter_scales) #Number of filter scales
    filt_num = 4 #Number of filters
    filt_bank = scale_size*filt_num
    
    #######Optimize code, use fewer loops

    # ----- TODO -----
    H,W = img.shape[0],img.shape[1]
    filter_responses = np.zeros((H,W,3*filt_bank))
    gauss_img = np.zeros((H,W,3))
    laplace_img = np.zeros((H,W,3))
    for loc  in range(0,scale_size):
        
        for layer in range(3):
            
            gauss_img = scipy.ndimage.gaussian_filter(lab_img[:,:,layer],filter_scales[loc]) #apply gaussian filter to the lab image
            laplace_img = scipy.ndimage.gaussian_laplace(lab_img[:,:,layer],filter_scales[loc]) #apply gaussian laplace filter to the lab image
            gauss_x = scipy.ndimage.gaussian_filter(lab_img[:,:,layer],filter_scales[loc],[1,0]) #apply first order gaussian filter in x to the lab image
            gauss_y = scipy.ndimage.gaussian_filter(lab_img[:,:,layer],filter_scales[loc],[0,1]) #apply first order gaussian filter in y to the lab image
            
            #Save filter responses
            #Improve saving approach
            filter_responses[:,:,loc*filt_num*3 + layer] = gauss_img
            filter_responses[:,:,loc*filt_num*3 + 3 + layer] = laplace_img
            filter_responses[:,:,loc*filt_num*3 + 6 + layer] = gauss_x
            filter_responses[:,:,loc*filt_num*3 + 9 + layer] = gauss_y
#       
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    opts = get_opts()
#    print(args)
    ind,alpha,train_files = args
    
#    # ----- TODO -----
#    #Inputs needed: location of the image, alpha (random pixels), image  path
    img = Image.open("../data/"+ (train_files)) #read an image
#
    img = np.array(img).astype(np.float32)/255
#
    filter_resp = extract_filter_responses(opts,img) #extract the responses
    rand_loc_y = np.random.choice(filter_resp.shape[0],int(alpha)) #Sample out alpha random pixels from the images
    rand_loc_x = np.random.choice(filter_resp.shape[1], int(alpha))
#    
    img_sub = filter_resp[rand_loc_y,rand_loc_x,:] #Extract the random pixels of size alpha*3*F
    np.save(os.path.join("../temp/", str(ind)+'.npy'), img_sub)

def compute_dictionary(opts, n_worker=6):
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
    alpha = opts.alpha
    
    train_files = open(join(data_dir, 'train_files_small.txt')).read().splitlines() #load the training data
    T_img = len(train_files)#size of training data--> # of images, T
    T_img_list = np.arange(T_img)
    alpha_list = np.ones(T_img)* alpha
    
    
    prcs = Pool(n_worker)
    args = list(zip(T_img_list,alpha_list,train_files))
    prcs.map(compute_dictionary_one_image,args)     #create subprocesses to  call ome_image
    
    # ----- TODO -----
    filter_responses = []
    for ind in range(0,T_img):
        tmp_file = np.load("../temp/"+str(ind)+'.npy') #load temp files back
        filter_responses.append(tmp_file)
        
    filter_responses = np.concatenate(filter_responses,axis=0)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)     #run k-means
    dictionary = kmeans.cluster_centers_

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

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
    H,W,_ = img.shape
    
    filt_img = extract_filter_responses(opts,img) #get filtered response
    filt_img = filt_img.reshape(H*W,dictionary.shape[-1])
    eucld = scipy.spatial.distance.cdist(filt_img,dictionary,'euclidean') #match each pixel with dictionary
    closest_word = np.argmin(eucld,axis=1) #use euclidian distance to find the closest match in the dictionary
    wordmap = closest_word.reshape(H,W) #matrix with the same width and height as img, where each pixel in wordmap is assigned the closest visual word of the filter response at the respective pixel in img
    print(closest_word.shape)
    return wordmap