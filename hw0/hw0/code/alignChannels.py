import numpy as np
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    
    disp_max  = 30 #maximum displacement
    
    #compute SSD to find error between two color intensities
    def ssdCalc(colr1,colr2):
        ssd_err = ((colr1-colr2)**2).sum()
        return ssd_err

#find the error of the original rgb matrices
#displace row and column to compare the error to original ssd error
for row in range(disp_max):
    for col in range(disp_max):
        row_shift = np.roll(colr2,i,axis=0)
        col_shift = np.roll(row_shift,j,axis=1)
        new_colr2 = col_shift
        err_disp = ssdCalc(colr1,new_colr2) #find the ssd error for the new shifted array
        
        if err_disp < 
    return None
