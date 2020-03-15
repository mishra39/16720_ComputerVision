"""Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import pdb
import matplotlib.pyplot as plt
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):

    # Scale the correspondence
    A = np.empty((pts1.shape[0],9))

    pts1 = pts1 / M
    pts2 = pts2 / M
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    T = np.array([[1/M, 0, 0],
                  [0, 1/M, 0],
                  [0,  0,  1]])

    # Construct A matrix
    A = np.vstack((x2 * x1, x2 * y1 , x2, y2 * x1,  y2 * y1, y2, x1, y1, np.ones(pts1.shape[0]))).T


    u, s, vh = np.linalg.svd(A) # Find SVD of AtA
    F = vh[-1].reshape(3,3) # Fundamental Matrix is column corresponding to the least singular values

    F = helper.refineF(F, pts1, pts2) # Refine F by using local minimization

    # Enforce rank2 constraint a.k.a singularity condition
    F = helper._singularize(F)

    # Unscale the fundamental matrix
    F = np.dot((np.dot(T.T,F)) , T)

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pass


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):

    E = np.dot((np.dot(K2.T ,F)), K1)

    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    w = np.empty((x1.shape[0],3))

    A1 = np.vstack([C1[0,0] - C1[2,0]*x1, C1[0,1] - C1[2,1]*x1, C1[0,2] - C1[2,2]*x1, C1[0,3] - C1[2,3]*x1]).T
    A2 = np.vstack([C1[1,0] - C1[2,0]*y1, C1[1,1] - C1[2,1]*y1, C1[1,2] - C1[2,2]*y1, C1[1,3] - C1[2,3]*y1]).T
    A3 = np.vstack([C2[0,0] - C2[2,0]*x2, C2[0,1] - C2[2,1]*x2, C2[0,2] - C2[2,2]*x2, C2[0,3] - C2[2,3]*x2]).T
    A4 = np.vstack([C2[1,0] - C2[2,0]*y2, C2[1,1] - C2[2,1]*y2, C2[1,2] - C2[2,2]*y2, C2[1,3] - C2[2,3]*y2]).T

    for i in range(x1.shape[0]):
        A = np.vstack((A1[i,:], A2[i,:], A3[i,:], A4[i,:]))
        u, s, vh = np.linalg.svd(A)
        wi_hom = vh[-1] # Nx4 vector
        wi  = wi_hom[0:3]/vh[-1,-1] # Nx3 vector
        w[i,:]  = wi

    w_hom = np.hstack((w,np.ones([w.shape[0],1])))

    # Reprojecting
    pts1_hat = np.dot(C1 , w_hom.T)
    pts2_hat = np.dot(C2 , w_hom.T)

    # pdb.set_trace()
    # Normalizing
    p1_hat_norm = (np.divide(pts1_hat[0:2,:] , pts1_hat[2,:])).T
    p2_hat_norm = (np.divide(pts2_hat[0:2,:] , pts2_hat[2,:])).T

    err1 = np.square(pts1[:,0] - p1_hat_norm[:,0]) + np.square(pts1[:,1] - p1_hat_norm[:,0])
    err2 = np.square(pts2[:,0] - p2_hat_norm[:,0]) + np.square(pts2[:,1] - p2_hat_norm[:,0])
    err = np.sum(err1) + np.sum(err2)

    return w,err
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):

    # Extract window from im1 around x1, y1
    rect_size = 10 # Size of the window

    # if np.abs(x1) >= rect_size//2 and np.abs(y1) >= rect_size//2:
    im1_sec = im1[(y1 - rect_size//2): np.rint(y1 + rect_size//2 + 1), (x1 - rect_size//2): (x1 + rect_size//2 + 1),:]  # Section of im1 around x1,y1


    im2_h, im2_w,_ = im2.shape # Size of im2
    # Ensure that the image section lies within the image dimensions

    pt1 = np.array([x1, y1, 1]) # homogeneous coordinates of im1
    ep_line = np.dot(F,pt1) # Epipolar Line
    ep2_l = ep_line / np.linalg.norm(ep_line)
    ep2_y = np.array(range(y1-rect_size//2,y1+rect_size//2)) # search coordinates for im2
    ep2_x = np.rint((ep2_l[1]*ep2_y + ep2_l[2])/ep2_l[0])

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres):
    # Replace pass by your implementation
    pass


if __name__ == "__main__":
    # 2.1
    pts = np.load('../data/some_corresp.npz')
    pts1 = pts['pts1']
    pts2 = pts['pts2']
    img1 = plt.imread('../data/im1.png')
    img2 = plt.imread('../data/im2.png')
    M = np.max(img1.shape)

    F = eightpoint(pts1, pts2, M) # EightPoint algrithm to find F
    np.savez('q2_1.npz', F, M)
    # helper.displayEpipolarF(img1, img2, F) # Visualize result

    # 3.1
    # import camera instrinsics
    K = np.load('../data/intrinsics.npz')
    K1 = K['K1']
    K2 = K['K2']
    E = essentialMatrix(F, K1, K2)
