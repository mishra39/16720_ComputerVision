'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import matplotlib.pyplot as plt
import numpy as np
import submission
import helper

M2_all = np.zeros([3,4,4])
pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']

K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']

img1 = plt.imread('../data/im1.png')
img2 = plt.imread('../data/im2.png')
M = np.max(img1.shape)

F = submission.eightpoint(pts1, pts2, M) # EightPoint algrithm to find F
E = submission.essentialMatrix(F, K1, K2)
M1 = np.eye(3)
M1 = np.hstack((M1, np.zeros([3,1])))
M2_all = helper.camera2(E)

C1 = np.dot(K1 , M1)
err_val = np.inf

for i in range(M2_all.shape[2]):

    C2 = np.dot(K2 , M2_all[:,:,i])
    w,err = submission.triangulate(C1, pts1, C2, pts2)

    if err < err_val:
        err_val = err
        M2 = M2_all[:,:,i]
        C2_best = C2
        w_best = w

np.savez('q3_3.npz', M2 = M2, C2 = C2_best, P = w_best)