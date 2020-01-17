import numpy as np

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    
    disp_max  = 30 #maximum displacement
    [rows,cols] = [red.shape[0],red.shape[1]]
    rgb_mat = np.zeros((rows,cols,3),'uint8') #Initialize matrix for imposing RGB layers on top of each other
    
    #compute SSD to find error between two color intensities
    def ssdCalc(colr1,colr2):
        ssd_err = ((colr1-colr2)**2).sum()
        
        return ssd_err
    
    #find the minimum error by trying out all the possible combinations
    def errorTest(colr1,colr2):
        min_row = 0
        min_col = 0
        errInit = ssdCalc(colr1,colr2) #find the error of the original matrices
        
        #displace row and column to compare the error to original ssd error
        for row in range(-disp_max,disp_max):
            for col in range(-disp_max,disp_max):
                new_colr2 = np.roll(colr2,[row,col],axis = [0,1])
                err_disp = ssdCalc(colr1,new_colr2) #find the ssd error for the new shifted array
                
                if err_disp < errInit:
                    errInit = err_disp
                    min_row = row
                    min_col = col
                    
        return [min_row,min_col]
    
    #function to update the values of the rgb matrix
    def rgbUpdate(colr1,colr2,colrIndex):
        [min_error_row,min_error_col] = errorTest(colr1,colr2)
        colr_new = np.roll(colr2,[min_error_row,min_error_col],axis = [0,1])
        rgb_mat[...,colrIndex] = colr_new
        
        return rgb_mat
    
    rgb_mat[...,0] = red
    rgb_mat = rgbUpdate(red,blue,2)
    rgb_mat = rgbUpdate(red,green,1)

    return rgb_mat