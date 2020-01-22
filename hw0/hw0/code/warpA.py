import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation.""" 
    
    #Determine the dimensions of input image to create an array of equal dimensions
    height = output_shape[0]
    width = output_shape[1]
    
    
    row,col = np.meshgrid(np.arange(height),np.arange(width),indexing='ij') #Create meshgrid
    row = row.flatten('F') #Reshape height dimension to a column
    col = col.flatten('F') #Reshape width dimension to a column
    
    P_i_warped = np.array([row,col,np.ones(row.size)]) #create an array with coordinates as grid
    
    P_i_src = np.round((np.linalg.inv(A)).dot(P_i_warped)) #Tranform the coordinates for the warped image to new locations
    
    x_cord,y_cord = [(P_i_src[0,:].astype(np.int32)),(P_i_src[1,:].astype(np.int32))] #Extract x and y coordinates for the warped image
    
    warp_im = np.zeros((height,width)) #zero matrix for storing warped image
    
    ind_x = np.logical_and(x_cord >= 0, x_cord < height) #Determine indices that are within the size of image
    
    ind_y = np.logical_and(y_cord >= 0, y_cord < width) #Determine indices that are within the size of image
    ind = np.logical_and(ind_x, ind_y) #Find the locations where both x and y coordinates satisfy the bounds
    
    x_cord, y_cord = x_cord[ind],y_cord[ind] #Update x and y coordinates
    row,col = row[ind],col[ind] #Find the same locations to create the warped image
    warp_im[row, col] = im[x_cord, y_cord]
    
    return warp_im

#for loop method
"""import numpy as npdef warp(im, A, output_shape):
    height = output_shape[0]
    width = output_shape[1]
    
    warp_im = np.zeros((height,width))
    
    The coordinates of the sampled output image points p_warped should be the rectangular
        range (0, 0) to (width − 1, height − 1) of integer values
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
    
    
    
    return warp_im"""
