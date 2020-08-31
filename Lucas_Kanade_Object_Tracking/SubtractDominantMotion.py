import numpy as np
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters) # LucasKanadeAffine(image1, image2, threshold, num_iters) #
    one_hom = np.array([[0, 0, 1]])
    M = np.append(M, one_hom, axis=0)

    im1_warp = affine_transform(image1, M, image1.shape)
    img_diff = np.abs(image2 - im1_warp)
    mask[img_diff < tolerance]  = 0
    return mask