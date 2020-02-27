import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    p = p0


    It_shape_0, It_shape_1 = It.shape
    It1_shape_0, It1_shape_1 = It1.shape

    # Interpolate the image
    It_spline = RectBivariateSpline(np.arange(It_shape_0), np.arange(It_shape_1),It)
    It1_spline = RectBivariateSpline(np.arange(It1_shape_0), np.arange(It1_shape_1),It1)

    # Rectangle Coordinates
    y_tl = rect[0]
    x_tl = rect[1]
    y_br = rect[2]
    x_br = rect[3]

    rect_h = int(y_br-y_tl+1)
    rect_w = int(x_br-x_tl+1)

    A_mat = np.zeros((rect_h*rect_w),2) # Matrix for gradients
    b_mat = np.zeros((rect_h*rect_w),1) # Matrix for change in pixel value difference between template and frame

    # initialize del_p ??????
    del_p = threshold + .2
    del_p = np.linalg.norm(del_p)

    # Create a meshgrid
    mesh_h, mesh_w = np.meshgrid(np.arange(rect_h),np.arange(rect_w))



    while del_p >= threshold:

        # Warping: Translation
        I_w_px = np.arange(x_tl + p[0], x_br + 0.1 + p[0]) # Translation in x
        I_w_py = np.arange(y_tl + p[1], y_br + 0.1 + p[1]) # Translation in y
        I_w_x = np.arange(x_tl, x_br + 0.1) # No Translation in x
        I_w_y = np.arange(y_tl, y_br + 0.1) # No Translation in y


        x_px, y_py = np.meshgrid(np.arange(I_w_px),np.arange(I_w_py)) # Create a meshgrid for new locations after translation
        xx, yy = np.meshgrid(np.arange(I_w_x),np.arange(I_w_y))

        # Jacobian for warp and gradient for image
        It1_dx = It1_spline.ev(y_py,x_px,dx=1).flatten()
        It1_dy = It1_spline.ev(y_py,x_px,dy=1).flatten()
        It_interp = It_spline.ev(yy,xx).flatten() # Warped values after interpolation
        It1_interp = It1_spline.ev(y_py,x_px).flatten()

        b_mat = It_interp - It1_interp
        A_mat[:,0] = It1_dx
        A_mat[:,1] = It1_dy

        H = np.dot(np.transpose(A_mat),A_mat)  #Hessian Calculation
        error = b
        del_p1 = np.dot(np.transpose(A),b)
        del_p = (np.dot(np.linalg.inv(H),del_p1)).flatten()
        p += del_p

    return p
