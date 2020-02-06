import os, math, multiprocessing
from os.path import join
from copy import copy
from opts import get_opts
import numpy as np
from PIL import Image
from multiprocessing import Pool
import visual_words
import matplotlib.pyplot as plt
import pdb

def get_feature_from_wordmap(opts, wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    # ----- TODO -----
    hist,_ = np.histogram(wordmap.flatten(),bins=np.arange(0,dict_size+1))
    
    hist = hist/np.sum(hist)
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    alpha = opts.alpha
    
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary) 
    
    # ----- TODO -----

#    H,W = wordmap.shape
#    H_rem = H % 16
#    W_rem = W % 16
#    a = 0
#    
#    if H_rem != 0:
#        H = H - H_rem
#        a = 1
#    
#    if W_rem != 0:
#        W = W - W_rem
#        a = 1
#    
#    if a == 1:
#        W = int(W)
#        H = int(H)
#        dim = (H, W)
#    
#        wordmap = np.resize(wordmap, dim)
        
    hist_all = np.array([]) #prelocate
    
    for l_num in range(L,-1,-1):

        sec_layer = pow(2,l_num) #number of sections in the layer
        if l_num > 1:
            layr_wt = (pow(2,l_num-L-1)) #weight of the layer while concatenating
            
        else:
            layr_wt = (pow(2,-L))
        
#        for row in range(0,sec_layer):
#            for col in range(0,sec_layer):
#                sub_sec = wordmap[row*sec_layer:(row+1)*sec_layer,col*sec_layer:(col+1)*sec_layer]
##                print(row*sec_layer,(row+1)*sec_layer,col*sec_layer,(col+1)*sec_layer)
#                hist  = get_feature_from_wordmap(opts,sub_sec,dict_size)
#                hist_all = np.hstack(hist*layr_wt)
#        hist_all = hist_all.flatten()
        sub_map = [sub_mat for r_sub in np.array_split(wordmap,sec_layer,axis=0) for sub_mat in np.array_split(r_sub,sec_layer,axis=1)] # Splitting of rows
#        for col_split in range(0,len(sub_sec_rows)):
#            sub_sec_cols = np.split(sub_sec_rows[col_split], sec_layer, axis=1)
#            12 16 12 16

#            for i in sub_sec_cols:
#                hist = get_feature_from_wordmap(opts,i,dict_size)
#                hist_all.append(hist)
        #plot all the histograms
        for tot_layr in range(sec_layer*sec_layer):
            
            hist = get_feature_from_wordmap(opts,sub_map[tot_layr],dict_size)#plot the histogram of the section
            hist_all = np.hstack(hist)
        
        hist_all = hist_all*layr_wt
            
    if np.sum(hist_all)>0:
        return hist_all / np.sum(hist_all)
    else:    
        return hist_all

def get_image_feature(args):#ind, img_path,label): #args): #opts, img_path, dictionary
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    # ----- TODO -----
    ind, img_path,label = args
    opts = get_opts()
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    dict_size = len(dictionary) # size of dictionary
    img = Image.open("../data/"+ (img_path)) #read an image
    img = np.array(img).astype(np.float32)/255 #convert to 0-1 range values
    wordmap = visual_words.get_visual_words(opts,img,dictionary) # find the wordmap for the image
    
    feat = get_feature_from_wordmap_SPM(opts,wordmap) # plot the histogram of Spatial Pyramids
    
    word_hist = get_feature_from_wordmap(opts,wordmap,dict_size) # Histogram for the whole image
    np.savez("../temp/"+ "train_"+str(ind)+".npz",feat = feat, label=label,allow_pickle = True)
    
    
def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    
    
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    dict_size = len(dictionary)
    
    T_img = len(train_files)#size of training data--> # of images, T
    T_img_list = np.arange(T_img)
    features = []
    
    # ----- TODO -----
#    for i in range(0,T_img):
#        get_image_feature(i, train_files[i],train_labels[i])
    prcs = Pool(n_worker)
    args = list(zip(T_img_list, train_files,train_labels))
    prcs.map(get_image_feature,args)     #create subprocesses to  call ome_image
#    features = np.empty(T_img*21,T_img*2)
    
    for i in range(0,T_img):
        tmp_file = np.load("../temp/"+"train_"+str(i)+".npz",allow_pickle=True) #load temp files back
        feat = tmp_file['feat']
        features.append(feat)
#    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')x
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255
#    
#    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    # ----- TODO -----
#    get_feature_from_wordmap(opts,wordmap,dict_size)
#    hist_all = get_feature_from_wordmap_SPM(opts,wordmap)
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
     )
    print('in recognition')
    print('L is', opts.L)
    print('K is', opts.K)
    print('alpha:', opts.alpha)
    print('filter_scales',opts.filter_scales)
#    per_acc = evaluate_recognition_system(opts,8)
#    print(per_acc)
def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    hist_min = np.minimum(word_hist,histograms) # find minimum of each corresponding bin
    hist_sim = np.sum(hist_min,axis=1) # find the histogram intersection similarity between word_hist and histograms

    return hist_sim
def evaluate_recognition_system(opts, n_worker=8):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    wrng = []

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    dict_size = len(dictionary)
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    test_file_num = len(test_files)
    
    # ----- TODO -----
    conf_mat = np.zeros((8,8),dtype=int)
    test_pred_labl = np.zeros(test_file_num,dtype = int)
    
    train_feats = trained_system['features']
    train_labels = trained_system['labels']
    
    for ind_f in range(test_file_num):
        img = Image.open("../data/"+ (test_files[ind_f])) # Load test images
        img = np.array(img).astype(np.float32)/255 #convert to 0-1 range values
        wordmap_test = visual_words.get_visual_words(opts,img,dictionary)
        feature_test = get_feature_from_wordmap_SPM(opts,wordmap_test)
        sim_dist = distance_to_set(feature_test,train_feats)
        labl_pred = train_labels[np.argmax(sim_dist)] #predicted label
        test_pred_labl[ind_f] = labl_pred
        labl_true = test_labels[ind_f]
        conf_mat[labl_true,labl_pred] += 1
        if labl_true != labl_pred:
            wrng.append(test_files[ind_f])
#        pdb.set_trace()
        
    per_acc = np.trace(conf_mat) / np.sum(conf_mat)
    print('in evaluation')
    print('L is', opts.L)
    print('K is', opts.K)
    print('alpha is', opts.alpha)
#    print(per_acc)
    return conf_mat,per_acc