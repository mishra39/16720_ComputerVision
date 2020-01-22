import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    height = output_shape[0]
    width = output_shape[1]
    
    warp_im = np.zeros((height,width))
    
    """The coordinates of the sampled output image points p_warped should be the rectangular
        range (0, 0) to (width − 1, height − 1) of integer values"""
    for row in range(0,height):
        for col in range(0,width):
            P_i_warped = np.array([row,col,1]) #coordiantes of sampled output image
            P_i_src = np.round((np.linalg.inv(A)).dot(P_i_warped))  #coordinates of the source. The points p_source must be chosen such that their image, Ap_i_source , transforms to this rectangle
            
            x_cord,y_cord = [int(P_i_src[0]),int(P_i_src[1])]#looking up the value of each of the destination pixels by sampling the original image at the computed p_i_source
            
            if x_cord < 0 or x_cord>=height:
                warp_im[row][col] = 0
            elif y_cord < 0 or y_cord >= width:
                warp_im[row][col] = 0
            else:
                warp_im[row][col] = im[x_cord][y_cord]
    
    
    
    return warp_im
