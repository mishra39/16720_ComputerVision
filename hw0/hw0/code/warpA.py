import numpy as np
def warp2(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation.""" 
    height = output_shape[0]
    width = output_shape[1]
    
    row,col = np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
    row = row.flatten('F')
    col = col.flatten('F')
    
    P_i_warped = np.array([row,col,np.ones(row.size)])
    #print(P_i_warped)
    P_i_src = np.round((np.linalg.inv(A)).dot(P_i_warped))
    x_cord,y_cord = [(P_i_src[0,:].astype(np.int32)),(P_i_src[1,:].astype(np.int32))]
    warp_im = np.zeros((height,width))
    ind_x = np.logical_and(x_cord >= 0, x_cord < height)
    ind_y = np.logical_and(y_cord >= 0, y_cord < width)
    ind = np.logical_and(ind_x, ind_y)
    x_cord, y_cord = x_cord[ind],y_cord[ind]
    row,col = row[ind],col[ind]
    print(row.shape,col.shape,x_cord.shape,y_cord.shape)
    warp_im[row, col] = im[x_cord, y_cord]
    print(im[x_cord, y_cord])
    print(warp_im[row, col])
    return warp_im
