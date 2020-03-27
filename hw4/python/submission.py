"""Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import pdb
import scipy.optimize
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
    f1 = vh[-1].reshape(3,3) # Fundamental Matrix is column corresponding to the least singular values
    f2 = vh[-2].reshape(3,3)

    detF_func = lambda coeff: np.linalg.det(coeff*f1 + (1-coeff)*f2)

    a0 = detF_func(0)
    a1 = 2*(detF_func(1) - detF_func(-1))/3 - (detF_func(2)-detF_func(-2))/12
    a2 = (detF_func(1) + detF_func(-1))/2 - a0
    a3 = (detF_func(1) - detF_func(-1))/2 - a1

    coeff_sol = np.roots([a3,a2,a1,a0])

    F_mat = [coeff*f1 + (1-coeff)*f2 for coeff in coeff_sol]

    F_mat = [helper.refineF(F, pts1, pts2) for F in F_mat]

    # Unscale the fundamental matrix
    F = [np.dot((np.dot(T.T,F)) , T) for F in F_mat]

    return F


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
    rect_size = 20 # Size of the window
    x1 = int(x1)
    y1 = int(y1)
    im1_sec = im1[(y1 - rect_size//2): (y1 + rect_size//2 + 1), (x1 - rect_size//2): (x1 + rect_size//2 + 1),:]  # Section of im1 around x1,y1

    im2_h, im2_w,_ = im2.shape # Size of im1

    pt1 = np.array([x1, y1, 1]) # homogeneous coordinates of im1

    ep_line = np.dot(F,pt1) # Epipolar Line
    ep2_l = ep_line / np.linalg.norm(ep_line)
    a,b,c = ep2_l

    ep2_y = np.arange(im2_h)
    ep2_x = np.rint(-(ep2_l[1]*ep2_y + ep2_l[2])/ep2_l[0])

    # Gaussian weight distribution about the center
    rect_vec = np.arange(-rect_size//2 , rect_size//2 + 1 ,1)
    rect_x, rect_y = np.meshgrid(rect_vec, rect_vec)
    std_dev = 7
    gauss_wt = np.dot( (np.exp(-((rect_x**2 + rect_y**2) / (2 * (std_dev**2))))),1)
    gauss_wt = gauss_wt / np.sqrt(2*np.pi*std_dev**2)
    gauss_wt = np.sum(gauss_wt)
    err_val = 1e4

    for y2 in range((y1 - rect_size//2), (y1 + rect_size//2 + 1)):
        x2 = int((-b*y2-c)/a)
        if (x2 >= rect_size//2 and x2 + rect_size//2 < im2_w and y2 >=rect_size//2 and y2 + rect_size//2 < im2_h):
            im2_sec = im2[y2-rect_size//2:y2+rect_size//2+1,x2-rect_size//2:x2+rect_size//2+1,:]
            err = np.linalg.norm((im1_sec-im2_sec)*gauss_wt)
            if err < err_val:
                err_val = err
                y2_best = y2
                x2_best = x2

    return x2_best, y2_best

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters, tol):
    max_inliers = -1
    F = np.empty([3,3]) #3 by 3 empty matrix for the best homograph

    rand_1 = np.empty([7,2])
    rand_2 = np.empty([7,2])
    max_inliers = -1

    pts1_hom = np.vstack((pts1.T,np.ones([1,pts1.shape[0]])))
    pts2_hom = np.vstack((pts2.T,np.ones([1,pts1.shape[0]])))

    for ind in range(nIters):
        print(ind)
        tot_inliers = 0
        ind_rand = np.random.choice(pts1.shape[0],7)

        rand_1 = pts1[ind_rand,:]
        rand_2 = pts2[ind_rand,:]

        F = eightpoint(rand_1, rand_2, M)
        pred_x2_hom = np.dot(F,pts1_hom)
        pred_x2  = pred_x2_hom / np.sqrt(np.sum(pred_x2_hom[:2,:]**2,axis=0))

        err = abs(np.sum(pts2_hom*pred_x2,axis=0))
        inliers_num = err < tol
        tot_inliers = inliers_num[inliers_num.T].shape[0]
        if tot_inliers > max_inliers:
            print('in if')
            bestF = F
            max_inliers = tot_inliers
            inliers = inliers_num[inliers_num.T]

    return bestF, inliers
'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    R = np.empty((3,3),dtype=float)
    theta = np.linalg.norm(r)

    if theta == 0:
        R = np.eye(3)

    else:
        u = r/theta
        u_x = np.array([[0, -u[2], u[1]],[u[2], 0, -u[0]],[-u[1],u[0],0]])
        R = np.eye(3)*np.cos(theta) +  (1-np.cos(theta))*(np.dot(u,u.T)) + u_x*np.sin(theta)

    return R
'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    zero_tol = 1e-2
    A = (R - R.T)/2
    rho = np.array(A[[2, 0, 1], [1, 2, 0]])[:, None]
    s = np.linalg.norm(rho)
    c = (np.trace(R) -1)/2

    if s < zero_tol and (c - 1) < zero_tol:
        r = np.zeros((3,1))
        return r

    elif s < zero_tol and (c+1) < zero_tol:
        v_tmp = R + np.eye(3)
        for i in range(R.shape[0]):
            if np.count_nonzero(v_tmp[:,i]) > 0:
                v = v_tmp[:,i]
                break

        u = v/np.linalg.norm(v)
        u_pi = u*np.pi

        if (np.linalg.norm(u_pi) == np.pi) and (u_pi[0,0] == u_pi[1,0] == 0 and u_pi[2,0] < 0) or (u_pi[0,0] == 0 and u_pi[1,0] < 0) or (u_pi[0,0] < 0):
            r = -u_pi
            return r

        else:
            r = u_pi
            return  r


    else:
        u = rho/s
        theta = np.arctan2(s,c)
        r = u*theta
        return r

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
    P = x[:-6].reshape(-1,3)
    r2 = x[-6:-3].reshape(3,1)
    t2 = x[-3:].reshape(3,1)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2,t2)).reshape(3,4) # Extrinsics of camera 2
    C1 = np.dot(K1,M1)
    C2 = np.dot(K2,M2)
    P_hom = np.vstack((P.T, np.ones((1,P.shape[0]))))

    p1_hat = np.zeros((2, P_hom.shape[1]))
    p2_hat = np.zeros((2, P_hom.shape[1]))

    x1_hom = np.dot(C1,P_hom)
    x2_hom = np.dot(C2,P_hom)

    p1_hat[0, :] = (x1_hom[0, :]/ x1_hom[2, :])
    p1_hat[1, :] = (x1_hom[1, :]/x1_hom[2, :])
    p2_hat[0,:] = (x2_hom[0, :]/x2_hom[2, :])
    p2_hat[1, :] = (x2_hom[1, :]/ x2_hom[2, :])
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T

    residuals = np.concatenate([(p1 - p1_hat).reshape(-1),(p2 - p2_hat).reshape(-1)])
    return residuals

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
    R2_0 = M2_init[:, 0:3]
    t2_0 = M2_init[:, 3]
    r2_0 = invRodrigues(R2_0)
    fun = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x))
    print('here2')
    x0 = P_init.flatten()
    x0 = np.append(x0, r2_0.flatten())
    x0 = np.append(x0, t2_0.flatten())

    x_opt, _ = scipy.optimize.leastsq(fun,x0)
    print('here4')
    P2 = x_opt[0:-6].reshape(-1,3)
    r2 = x_opt[-6:-3].reshape(3,1)
    t2 = x_opt[-3:].reshape(3,1)

    R2 = rodrigues(r2)

    M2 = np.hstack((R2,t2)) # Extrinsics of camera 2
    return M2,P2

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

    P12, err12 = triangulate(C1, pts1[:,:2], C2, pts2[:,:2])
    # P23, err23 = triangulate(C2, pts2[:,:2], C3, pts3[:,:2])
    # P13, err13 = triangulate(C1, pts1[:,:2], C2, pts3[:,:2])

    return P12, err12


if __name__ == "__main__":

    # # 2.1
    # pts = np.load('../data/some_corresp.npz')
    # pts1 = pts['pts1']
    # pts2 = pts['pts2']
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)

    # F = sevenpoint(pts1[1:7,:], pts2[1:7,:], M) # EightPoint algrithm to find F

    # F = eightpoint(pts1, pts2, M) # EightPoint algrithm to find F
    # np.savez('q2_1.npz', F=F, M=M)
    # helper.displayEpipolarF(im1, im2, F) # Visualize result

    # # 3.1
    # import camera instrinsics
    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']
    # E = essentialMatrix(F, K1, K2)

    # 4.1
    # x1 = pts1[10,0]
    # y1 = pts1[10,1]
    # x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
    # sel_pts1 , sel_pts2 = helper.epipolarMatchGUI(im1, im2, F)
    # np.savez('q4_1.npz', F = F, pts1 = sel_pts1, pts2 = sel_pts2)

    # # 4.2


    # # 5.1
    # pts = np.load('../data/some_corresp_noisy.npz')
    # pts1 = pts['pts1']
    # pts2 = pts['pts2']
    # im1 = plt.imread('../data/im1.png')
    # im2 = plt.imread('../data/im2.png')
    # M = np.max(im1.shape)
    # nIters = 10
    # tol = 1
    # F,inliers = ransacF(pts1, pts2, M, nIters, tol)
    # helper.displayEpipolarF(im1,im2,F)

    # # 5.3

    # M1 = np.eye(3)
    # M1 = np.hstack((M1, np.zeros([3,1])))
    # E = essentialMatrix(F, K1, K2)
    # M2_all = helper.camera2(E)

    # C1 = np.dot(K1 , M1)
    # err_val = np.inf

    # for i in range(M2_all.shape[2]):

    #     C2 = np.dot(K2 , M2_all[:,:,i])
    #     w,err = triangulate(C1, pts1, C2, pts2)

    #     if err < err_val:
    #         err_val = err
    #         M2 = M2_all[:,:,i]
    #         C2_best = C2
    #         w_best = w

    # P_init,err = triangulate(C1, pts1, C2_best, pts2)
    # M2_out, P2 = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_init)

    # 6.1
    time_0 = np.load('../data/q6/time0.npz')
    pts1 = time_0['pts1'] # Nx3 matrix
    pts2 = time_0['pts2'] # Nx3 matrix
    pts3 = time_0['pts3'] # Nx3 matrix
    M1_0 = time_0['M1']
    M2_0 = time_0['M2']
    M3_0 = time_0['M3']
    K1_0 = time_0['K1']
    K2_0 = time_0['K2']
    K3_0 = time_0['K3']
    C1_0 = np.dot(K1_0,M1_0)
    C2_0 = np.dot(K1_0,M2_0)
    C3_0 = np.dot(K1_0,M3_0)
    Thres = 575
    P_mv, err_mv = MultiviewReconstruction(C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)
    M2_opt,P2_opt = bundleAdjustment(K2_0, M2_0, pts2[:,:2], K3_0, M3_0, pts3[:,:2], P_mv)
    helper.plot_3d_keypoint(P2_opt)

    # #bundleAdjustment
    # R2_0 = M2_init[:, 0:3]
    # t2_0 = M2_init[:, 3]
    # r2_0 = invRodrigues(R2_0)
    # fun = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x))
    # print('here2')
    # x0 = P_init.flatten()
    # x0 = np.append(x0, r2_0.flatten())
    # x0 = np.append(x0, t2_0.flatten())

    # x_opt, _ = scipy.optimize.leastsq(fun,x0)
    # print('here4')
    # P2 = x_opt[0:-6].reshape(-1,3)
    # r2 = x_opt[-6:-3].reshape(3,1)
    # t2 = x_opt[-3:].reshape(3,1)

    # R2 = rodrigues(r2)

    # M2 = np.hstack((R2,t2)) # Extrinsics of camera 2
    # return M2,P2
    # # M2_out, P2 = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_init)
    # img = plt.imread('../data/q6/cam1_time0.jpg')
    # helper.visualize_keypoints(img, pts1, 575)



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(P2[:,0],P2[:,1],P2[:,2],c='b')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()
    # plt.savefig('fig_5_3.jpg')