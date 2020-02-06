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


def main():
    opts = get_opts()
    print('L is', opts.L)
    print('K is', opts.K)
    print('alpha is', opts.alpha)
    print()
#     Q1.1
#    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255
#    filter_responses = visual_words.extract_filter_responses(opts, img)
##    imageio.imsave('../results/filter_responses.jpg',filter_responses)
#    util.display_filter_responses(opts, filter_responses)
#
##    # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
###
#    ## Q1.3
#    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255
#    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    util.visualize_wordmap(img)
#    util.visualize_wordmap(wordmap)
##    
#    img_path = join(opts.data_dir, 'waterfall/sun_bbeqjdnienanmmif.jpg')
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255
#    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    util.visualize_wordmap(img)
#    util.visualize_wordmap(wordmap)
##    
#    img_path = join(opts.data_dir, 'windmill/sun_bratfupeyvlazpba.jpg')
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255  
#    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    util.visualize_wordmap(img)
#    util.visualize_wordmap(wordmap)
#    
#    img_path = join(opts.data_dir, 'desert/sun_adjlepvuitklskrz.jpg')
#    img = Image.open(img_path)
#    img = np.array(img).astype(np.float32)/255  
#    wordmap = visual_words.get_visual_words(opts, img, dictionary)
#    util.visualize_wordmap(img)
#    util.visualize_wordmap(wordmap)
    
    
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


if __name__ == '__main__':
    start = time.time()
    main()
    # python 3
    print('It took', ((time.time()-start)/60), 'minutes to execute the program.')