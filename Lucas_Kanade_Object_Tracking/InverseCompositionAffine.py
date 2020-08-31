import numpy as np
from scipy.ndimage import affine_transform
def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Compute Image gradient
    I_dy, I_dx = np.gradient(It1)

    iters = 0


    while True and iters <= num_iters:

        iters = iters + 1
        M_copy = np.copy(M)
        M_copy[0:2, 0:2] = np.fliplr(M_copy[0:2, 0:2])
        M_copy = np.flipud(M_copy)
        print(iters)

        M_copy = np.append(M_copy, np.array([[0,0,1]]), axis = 0)
        # Applying affine warp
        It1_warp = affine_transform(It1, M_copy, cval = -7)
        warp_points = np.where(It1_warp !=-7)

        b = It1_warp[warp_points] - It[warp_points]

        A = np.zeros([b.size,6])

        A[:,0] = I_dx[warp_points] * warp_points[1]
        A[:,1] = I_dx[warp_points] * warp_points[0]
        A[:,2] = I_dx[warp_points]

        A[:,3] = I_dy[warp_points] * warp_points[1]
        A[:,4] = I_dy[warp_points] * warp_points[0]
        A[:,5] = I_dy[warp_points]

        del_p = np.linalg.lstsq(A,b, rcond = -1)
        del_p = del_p[0]
        del_p_norm = np.linalg.norm(del_p)

        if del_p_norm < threshold:
            break;
        else:
            del_p2 = del_p.reshape(2,3)
            del_p2[0][0] = 1 + del_p2[0][0]
            del_p2[1][1] = 1 + del_p2[1][1]
            del_p2 = np.append(del_p2, np.array([[0,0,1]]), axis = 0)
            del_M = np.append(M, np.array([[0,0,1]]), axis = 0)
            del_M = del_M @ np.linalg.inv(del_p2)
            M = del_M[0:2,:]
    return M