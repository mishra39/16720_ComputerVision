from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import util
import visual_words
import visual_recog
from opts import get_opts
import pdb
import time

filter_scales = np.array([[1, 2, 5, 10, 25, 50]])
K = [10,77,77,77,100]
alpha = [25,25,75,75,75]
L  = [2,2,2,4,4]    

def main():
    
    for ind in range(len(K)):
        start = time.time()
        print('Started at: ',start)
        opts = get_opts()
        opts.L = L[ind]
        opts.K = K[ind]
        opts.alpha = alpha[ind]
        print('filter_scales',opts.filter_scales)
        print('L is', opts.L)
        print('K is', opts.K)
        print('alpha is', opts.alpha)
    
        n_cpu = util.get_num_CPU()
        visual_words.compute_dictionary(opts, n_worker=n_cpu)
        dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
        
        # Q2.1-2.4
        n_cpu = util.get_num_CPU()
        visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    
        ## Q2.5
        n_cpu = util.get_num_CPU()
        conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
        
        print(conf)
        print(accuracy)
        np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
        np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')
        print('Finished at: ',time.time())
        print('It took', ((time.time()-start)/60), 'minutes to execute the iteration.')
if __name__ == '__main__':
    start = time.time()
    main()
    # python 3
    print('Total execution time:', ((time.time()-start)/60), 'minutes.')
    