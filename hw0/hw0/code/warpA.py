import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    length = output_shape[0]
    height = output_shape[1]
    lx = np.linspace(1,length)
    hy = np.linspace(1,height)
    [cx,cy] = np.meshgrid(lx,hy)
    mat_len = np.size(lx)
    mat_leny = np.size(hy)
    lx = np.reshape(mat_len,1)
    hy = np.reshape(mat_leny,1)
    
    Pw = np.array()
    return None
